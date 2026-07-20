// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Live server-streaming plumbing: the broadcast→gRPC adapter used by
//! `SubscribeMetrics`, and the `SubscribeFilter` stream that prepends a
//! per-model full snapshot before joining the live feed.

use std::pin::Pin;
use std::sync::Arc;

use async_stream::stream;
use futures::Stream;
use tokio::sync::broadcast;
use tokio_util::sync::CancellationToken;
use tonic::Status;

use dynamo_kv_event_relay_proto::wire::images::encode_snapshot_chunks;
use dynamo_kv_event_relay_proto::{FilterUpdate, MetricsSnapshot};

use crate::clock::unix_micros;
use crate::filter::{FilterFrame, FilterSnapshot};
use crate::observability::RelayMetrics;

pub(crate) type SubscribeMetricsStream =
    Pin<Box<dyn Stream<Item = Result<MetricsSnapshot, Status>> + Send + 'static>>;
pub(crate) type SubscribeFilterStream =
    Pin<Box<dyn Stream<Item = Result<FilterUpdate, Status>> + Send + 'static>>;

/// Frames carry the relay `instance_id` only on the first frame of each stream
/// (it is constant per stream); broadcast-built frames have it empty.
pub(crate) trait StampInstanceId {
    fn stamp_instance_id(&mut self, id: &bytes::Bytes);
}

impl StampInstanceId for MetricsSnapshot {
    fn stamp_instance_id(&mut self, id: &bytes::Bytes) {
        self.instance_id = id.clone();
    }
}

/// Adapter: take a `broadcast::Receiver<T>` and emit gRPC stream items.
/// `Lagged` is fatal for the stream so the client reconnects and establishes
/// a new base from the snapshots prepended to the replacement stream.
pub(crate) fn broadcast_to_stream<T>(
    mut rx: broadcast::Receiver<T>,
    channel: &'static str,
    subscriber_id: String,
    metrics: Option<Arc<RelayMetrics>>,
    instance_id: bytes::Bytes,
    cancel: CancellationToken,
) -> impl Stream<Item = Result<T, Status>> + Send + 'static
where
    T: Clone + Send + StampInstanceId + 'static,
{
    stream! {
        // RAII: bumps `active_subscribers{channel}` now, decrements when
        // this stream is dropped (client disconnect or graceful end).
        let _guard = metrics.as_ref().map(|m| m.subscriber_guard(channel));
        let mut first = true;
        loop {
            // End promptly on shutdown. The broadcast sender lives for
            // the process lifetime, so `recv()` never returns `Closed`; without
            // this the stream blocks graceful shutdown until the k8s SIGKILL.
            let recv = tokio::select! {
                biased;
                _ = cancel.cancelled() => {
                    yield Err(Status::unavailable("relay shutting down"));
                    break;
                }
                r = rx.recv() => r,
            };
            match recv {
                Ok(mut item) => {
                    if first {
                        item.stamp_instance_id(&instance_id);
                        first = false;
                    }
                    yield Ok(item)
                }
                Err(broadcast::error::RecvError::Closed) => {
                    tracing::info!(
                        channel,
                        subscriber_id = %subscriber_id,
                        "broadcast closed, ending stream"
                    );
                    break;
                }
                Err(broadcast::error::RecvError::Lagged(n)) => {
                    if let Some(m) = &metrics {
                        m.subscriber_lagged_total.with_label_values(&[channel]).inc();
                    }
                    tracing::warn!(
                        channel,
                        subscriber_id = %subscriber_id,
                        skipped = n,
                        "subscriber lagged, closing stream for replay recovery"
                    );
                    yield Err(Status::resource_exhausted(format!(
                        "subscriber lagged on {channel}; skipped {n} frames"
                    )));
                    break;
                }
            }
        }
    }
}

/// Build one `SubscribeFilter` wire frame, stamping the relay `instance_id`
/// only on the first frame of the stream (it is constant per stream).
fn filter_update(
    model_key: u64,
    seq: u64,
    send_ts_us: u64,
    payload: bytes::Bytes,
    heartbeat: bool,
    instance_id: &bytes::Bytes,
    first: &mut bool,
) -> FilterUpdate {
    let stamp = std::mem::take(first);
    FilterUpdate {
        seq,
        send_ts_us,
        model_key,
        payload,
        instance_id: if stamp {
            instance_id.clone()
        } else {
            bytes::Bytes::new()
        },
        heartbeat,
    }
}

/// `SubscribeFilter` stream: send each model's current full snapshot (chunked)
/// before joining the live
/// broadcast. A live full snapshot arrives as a single [`FilterFrame::Full`]
/// carrying a copy-on-write `Arc` to the lane words — the CBI1 chunks are
/// serialized here, per subscriber, so the publisher never does that memcpy
/// under its lock. `Lagged` closes the stream; the client reconnects and
/// re-syncs from a fresh snapshot.
pub(crate) fn filter_subscribe_stream(
    snapshots: Vec<FilterSnapshot>,
    rx: broadcast::Receiver<FilterFrame>,
    instance_id: bytes::Bytes,
    metrics: Option<Arc<RelayMetrics>>,
    subscriber_id: String,
    cancel: CancellationToken,
) -> impl Stream<Item = Result<FilterUpdate, Status>> + Send + 'static {
    stream! {
        let _guard = metrics.as_ref().map(|m| m.subscriber_guard("filter"));
        let mut first = true;
        let now = unix_micros();
        for snapshot in snapshots {
            for chunk in encode_snapshot_chunks(snapshot.format, 0, snapshot.epoch, &snapshot.words) {
                yield Ok(filter_update(snapshot.model_key, 0, now, chunk.into(), false, &instance_id, &mut first));
            }
        }
        let mut rx = rx;
        loop {
            // End promptly on shutdown (see broadcast_to_stream).
            let recv = tokio::select! {
                biased;
                _ = cancel.cancelled() => {
                    yield Err(Status::unavailable("relay shutting down"));
                    break;
                }
                r = rx.recv() => r,
            };
            match recv {
                Ok(FilterFrame::Full { model_key, seq, send_ts_us, format, epoch, words }) => {
                    for chunk in encode_snapshot_chunks(format, 0, epoch, &words) {
                        yield Ok(filter_update(model_key, seq, send_ts_us, chunk.into(), false, &instance_id, &mut first));
                    }
                }
                Ok(FilterFrame::Delta { model_key, seq, send_ts_us, bytes }) => {
                    yield Ok(filter_update(model_key, seq, send_ts_us, bytes, false, &instance_id, &mut first));
                }
                Ok(FilterFrame::Heartbeat { model_key, seq, send_ts_us }) => {
                    yield Ok(filter_update(model_key, seq, send_ts_us, bytes::Bytes::new(), true, &instance_id, &mut first));
                }
                Err(broadcast::error::RecvError::Closed) => break,
                Err(broadcast::error::RecvError::Lagged(n)) => {
                    if let Some(m) = &metrics {
                        m.subscriber_lagged_total.with_label_values(&["filter"]).inc();
                    }
                    tracing::warn!(
                        subscriber_id = %subscriber_id,
                        skipped = n,
                        "filter subscriber lagged; closing stream (reconnect re-syncs from snapshot)"
                    );
                    yield Err(Status::resource_exhausted(format!(
                        "filter subscriber lagged; skipped {n} frames"
                    )));
                    break;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dynamo_kv_event_relay_proto::wire::images::{
        FilterFormat, ImagesFrame, SnapshotAssembly, decode,
    };
    use futures::StreamExt;

    // A live `FilterFrame::Full` must expand, in the subscriber stream, into
    // CBI1 snapshot chunks the consumer reassembles back to the exact lane
    // words, with seq/model_key carried through and instance_id on the first
    // frame only.
    #[tokio::test]
    async fn full_frame_expands_into_reassemblable_chunks() {
        let format = FilterFormat::new(0x5EED, 1 << 12).unwrap();
        let mut words = vec![0u64; format.bucket_count];
        words[1] = 0xABCD;
        words[100] = 0x1234_5678;
        words[format.bucket_count - 1] = 0xFFFF;
        let words: Arc<[u64]> = Arc::from(words);

        let (tx, rx) = broadcast::channel(16);
        let stream = filter_subscribe_stream(
            Vec::new(),
            rx,
            bytes::Bytes::from_static(b"relay-1"),
            None,
            "sub".into(),
            CancellationToken::new(),
        );
        tokio::pin!(stream);

        let sent = tx.send(FilterFrame::Full {
            model_key: 7,
            seq: 5,
            send_ts_us: 99,
            format,
            epoch: 3,
            words: words.clone(),
        });
        assert!(sent.is_ok(), "broadcast has a live receiver");
        drop(tx); // close the broadcast so the stream ends after draining

        let mut assembly = SnapshotAssembly::new(format);
        let mut assembled = None;
        let mut first = true;
        while let Some(item) = stream.next().await {
            let update = item.expect("frame ok");
            assert_eq!(update.model_key, 7);
            assert_eq!(update.seq, 5);
            if std::mem::take(&mut first) {
                assert_eq!(update.instance_id.as_ref(), b"relay-1");
            } else {
                assert!(update.instance_id.is_empty());
            }
            let frame = decode(format, &update.payload).unwrap();
            assert!(matches!(frame, ImagesFrame::SnapshotChunk { .. }));
            if let Some(done) = assembly.absorb(&frame).unwrap() {
                assembled = Some(done);
            }
        }
        let (epoch, images) = assembled.expect("full snapshot reassembled from stream chunks");
        assert_eq!(epoch, 3);
        assert_eq!(images.len(), 3, "three nonzero buckets");
    }
}
