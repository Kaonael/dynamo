// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;
use std::time::Duration;

use parking_lot::RwLock;
use tokio::sync::{broadcast, watch};
use tokio_util::sync::CancellationToken;

use super::protocol as proto;

/// A complete load window published by [`run_snapshot_publisher`].
pub(super) trait LoadWindow: Clone + Send + Sync + 'static {
    fn window_sequence(&self) -> u64;
}

impl LoadWindow for proto::LoadUpdate {
    fn window_sequence(&self) -> u64 {
        self.window_sequence
    }
}

/// Latest window plus bounded fanout of the following ones.
#[derive(Clone)]
pub(super) struct LoadUpdateHub<T> {
    updates: broadcast::Sender<T>,
    /// `None` until the first window, so no subscriber sees a placeholder
    /// that would read as every pool gone.
    current: Arc<RwLock<Option<T>>>,
}

impl<T: LoadWindow> LoadUpdateHub<T> {
    pub(super) fn new(capacity: usize) -> Self {
        let (updates, _) = broadcast::channel(capacity);
        Self {
            updates,
            current: Arc::new(RwLock::new(None)),
        }
    }

    pub(super) fn subscribe(&self) -> broadcast::Receiver<T> {
        self.updates.subscribe()
    }

    pub(super) fn current(&self) -> Option<T> {
        self.current.read().clone()
    }

    fn publish(&self, update: T) {
        *self.current.write() = Some(update.clone());
        let _ = self.updates.send(update);
    }
}

/// Publishes `window_update(snapshot, sequence)` for every snapshot `source`
/// produces, starting at sequence 1, so windows follow the source's cadence.
/// Fails once the source's sender is dropped, unless already cancelled.
pub(super) async fn run_snapshot_publisher<S: Clone, T: LoadWindow>(
    mut source: watch::Receiver<Option<S>>,
    updates: LoadUpdateHub<T>,
    cancel: CancellationToken,
    mut window_update: impl FnMut(&S, u64) -> T,
) -> anyhow::Result<()> {
    let mut sequence = 0u64;
    loop {
        let changed = tokio::select! {
            biased;
            _ = cancel.cancelled() => return Ok(()),
            changed = source.changed() => changed,
        };
        if changed.is_err() {
            if cancel.is_cancelled() {
                return Ok(());
            }
            anyhow::bail!("Pool Relay load source stopped");
        }
        // Clone out of the borrow so the source's sender is not blocked while
        // the window is built.
        let Some(snapshot) = source.borrow_and_update().clone() else {
            continue;
        };
        sequence = next_sequence(sequence)?;
        updates.publish(window_update(&snapshot, sequence));
    }
}

fn next_sequence(sequence: u64) -> anyhow::Result<u64> {
    sequence
        .checked_add(1)
        .ok_or_else(|| anyhow::anyhow!("Pool Relay load window sequence exhausted"))
}

pub(super) fn window_ms(window: Duration) -> u64 {
    u64::try_from(window.as_millis()).unwrap_or(u64::MAX)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// One window per snapshot, none before the first; a stopped source
    /// fails the publisher unless it was cancelled.
    #[tokio::test]
    async fn snapshot_publisher_follows_its_source() {
        let hub = LoadUpdateHub::new(4);
        assert!(hub.current().is_none());
        let (source, receiver) = watch::channel(None::<()>);
        let publisher = tokio::spawn(run_snapshot_publisher(
            receiver,
            hub.clone(),
            CancellationToken::new(),
            |_, sequence| proto::LoadUpdate {
                window_sequence: sequence,
                ..Default::default()
            },
        ));
        let mut updates = hub.subscribe();
        source.send_replace(Some(()));
        assert_eq!(updates.recv().await.unwrap().window_sequence, 1);
        source.send_replace(Some(()));
        assert_eq!(updates.recv().await.unwrap().window_sequence, 2);
        drop(source);
        let error = publisher.await.unwrap().unwrap_err();
        assert!(error.to_string().contains("stopped"));
    }
}
