// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::overlap::filter_overlap;
use super::producer::{DELTA_ENTRY_BYTES, MAX_DELTA_BYTES, churn_wants_full};
use super::snapshot::{CHUNK_BODY_PREFIX, parse_chunk};
use super::*;

fn spread(value: u64) -> u64 {
    value.wrapping_mul(0x9E37_79B9_7F4A_7C15)
}

/// Keep the argmax search aligned with the brute-force reference so pruning and
/// tie-breaking cannot drift.
#[test]
fn argmax_overlap_dc_matches_bruteforce() {
    let seed = DEFAULT_FILTER_SEED;
    let shared: Vec<u64> = (0..6).map(spread).collect();
    let num_dcs = 5usize;
    let mut filters = Vec::new();
    for dc in 0..num_dcs {
        let mut f = CuckooFilter::with_capacity_seeded(4096, seed);
        for &h in &shared {
            f.insert(h);
        }
        for k in 0..(dc as u64) * 4 {
            f.insert(spread(0x1000 + dc as u64 * 1000 + k));
        }
        filters.push(f);
    }
    let mut chain = shared.clone();
    for k in 0..16u64 {
        chain.push(spread(0x1000 + 4 * 1000 + k));
    }
    let probes = probes_for(&chain, seed);

    let (adc, adepth) = argmax_overlap_dc(&filters, &probes);
    let (mut bdc, mut bdepth) = (0usize, 0u32);
    for (dc, f) in filters.iter().enumerate() {
        let d = filter_overlap(f, &probes);
        if d > bdepth {
            bdepth = d;
            bdc = dc;
        }
    }
    assert_eq!((adc, adepth), (bdc, bdepth), "argmax/depth must match");
    assert!(adepth >= 6, "shared head (6) must at least match");

    assert_eq!(argmax_overlap_dc(&filters, &[]), (0, 0));
    let absent = probes_for(&[spread(1 << 40), spread((1 << 40) + 1)], seed);
    assert_eq!(argmax_overlap_dc(&filters, &absent).1, 0);
}

/// Keep the logarithmic search equivalent to the linear reference so routing
/// scores do not drift.
#[test]
fn overlap_depth_searched_matches_linear() {
    let seed = DEFAULT_FILTER_SEED;
    let mut filter = CuckooFilter::with_capacity_seeded(8192, seed);
    let chain: Vec<u64> = (0..100).map(spread).collect();
    for &h in &chain {
        filter.insert(h);
    }
    for k in 0..=100usize {
        let mut q: Vec<u64> = chain[..k].to_vec();
        q.push(spread(1u64 << 50)); // never inserted -> reliable miss
        q.push(spread((1u64 << 50) + 1));
        let probes = probes_for(&q, seed);
        assert_eq!(
            overlap_depth_searched(&filter, &probes),
            filter_overlap(&filter, &probes),
            "searched vs linear depth mismatch at k={k}"
        );
    }
}

/// Keep lazy probe derivation behaviorally identical to the precomputed probe
/// path.
#[test]
fn overlap_depth_searched_seq_matches_probe_array() {
    let seed = DEFAULT_FILTER_SEED;
    let mut filter = CuckooFilter::with_capacity_seeded(8192, seed);
    let chain: Vec<u64> = (0..100).map(spread).collect();
    for &h in &chain {
        filter.insert(h);
    }
    for k in 0..=100usize {
        let mut q: Vec<u64> = chain[..k].to_vec();
        q.push(spread(1u64 << 50));
        q.push(spread((1u64 << 50) + 1));
        assert_eq!(
            overlap_depth_searched_seq(&filter, &q),
            overlap_depth_searched(&filter, &probes_for(&q, seed)),
            "seq-driven vs probe-array depth mismatch at k={k}"
        );
    }
}

fn delta_bucket_count(bytes: &[u8]) -> u32 {
    u32::from_le_bytes(
        bytes[SNAP_HEADER_LEN + 8..SNAP_HEADER_LEN + 12]
            .try_into()
            .unwrap(),
    )
}

fn assemble_full(state: &SnapshotState) -> (CuckooFilter, SnapshotMeta) {
    assemble_chunks(state.chunks()).expect("assemble full snapshot")
}

fn expect_full(publish: Publish) -> SnapshotState {
    match publish {
        Publish::Full(state) => state,
        Publish::Delta(_) => panic!("expected full snapshot, got delta"),
        Publish::Unchanged => panic!("expected full snapshot, got unchanged"),
    }
}

fn expect_delta(publish: Publish) -> Vec<u8> {
    match publish {
        Publish::Delta(delta) => delta,
        Publish::Full(_) => panic!("expected delta, got full snapshot"),
        Publish::Unchanged => panic!("expected delta, got unchanged"),
    }
}

#[test]
fn ckf1_header_contract_is_stable() {
    let mut producer = SnapshotProducer::new(7, 1, DEFAULT_FILTER_SEED);
    let full = expect_full(producer.publish());
    let chunk = full.chunks().next().expect("one empty-filter chunk");

    assert_eq!(&chunk[0..4], b"CKF1");
    assert_eq!(&chunk[4..12], &[1, 0, 2, 0, 16, 4, 0, 0]);
    assert_eq!(&chunk[12..20], &DEFAULT_FILTER_SEED.to_le_bytes());
    assert_eq!(&chunk[20..28], &2u64.to_le_bytes());
    assert_eq!(&chunk[28..36], &7u64.to_le_bytes());
    assert_eq!(&chunk[36..44], &1u64.to_le_bytes());
}

#[test]
fn failed_insert_is_transactional() {
    for seed in 1..32 {
        let mut filter = CuckooFilter::with_capacity_seeded(1, seed);
        let mut inserted = Vec::new();
        for h in 0..10_000u64 {
            let before = filter.to_raw_buckets();
            let before_len = filter.len();
            if filter.insert(h.wrapping_mul(0x9E37_79B9_7F4A_7C15)) {
                inserted.push(h.wrapping_mul(0x9E37_79B9_7F4A_7C15));
                continue;
            }
            assert_eq!(filter.to_raw_buckets(), before);
            assert_eq!(filter.len(), before_len);
            assert!(inserted.iter().all(|&existing| filter.contains(existing)));
            break;
        }
    }
}

#[test]
fn cloned_filter_mutates_with_page_level_copy_on_write() {
    let mut original = CuckooFilter::provisioned(100_000, DEFAULT_FILTER_SEED);
    for hash in 0..10_000u64 {
        assert!(original.insert(spread(hash)));
    }
    let mut cloned = original.clone();
    let added = spread(1_000_000);
    assert!(cloned.insert(added));

    assert!(cloned.contains(added));
    assert!(!original.contains(added));
    assert_ne!(cloned.to_raw_buckets(), original.to_raw_buckets());
}

#[test]
fn snapshot_pages_remain_point_in_time_after_producer_mutation() {
    let mut producer = SnapshotProducer::new(7, 100_000, DEFAULT_FILTER_SEED);
    for hash in 0..10_000u64 {
        assert!(producer.insert(spread(hash)));
    }
    let snapshot = producer.current_snapshot();
    let added = spread(1_000_000);
    assert!(producer.insert(added));

    let (captured, _) = assemble_full(&snapshot);
    assert!(!captured.contains(added));
    let (current, _) = assemble_full(&producer.current_snapshot());
    assert!(current.contains(added));
}

#[test]
fn producer_delta_tracks_only_dirty_buckets_and_cancellations() {
    let mut producer = SnapshotProducer::new(7, 1024, DEFAULT_FILTER_SEED);
    let full = expect_full(producer.publish());
    let (mut consumer, meta) = assemble_full(&full);

    assert!(producer.insert(11));
    let delta = expect_delta(producer.publish());
    assert!(is_delta(&delta));
    assert!(delta_bucket_count(&delta) <= 2);
    apply_delta(&mut consumer, meta.filter_epoch, &delta).unwrap();
    assert_eq!(consumer.len(), 1);

    assert!(producer.insert(22));
    producer.remove(22);
    assert!(
        matches!(producer.publish(), Publish::Unchanged),
        "net-zero update must not publish"
    );

    producer.remove(11);
    let delta = expect_delta(producer.publish());
    apply_delta(&mut consumer, meta.filter_epoch + 1, &delta).unwrap();
    assert_eq!(consumer.len(), 0);
}

#[test]
fn producer_resize_forces_full_snapshot() {
    let mut producer = SnapshotProducer::new(9, 4, DEFAULT_FILTER_SEED);
    expect_full(producer.publish());
    producer.rebuild(0..100, 100);
    let state = expect_full(producer.publish());
    let (filter, _) = assemble_full(&state);
    assert_eq!(filter.len(), 100);
}

#[test]
fn mass_eviction_republishes_full_and_converges_stale_consumer() {
    let mut producer = SnapshotProducer::new(5, 1024, DEFAULT_FILTER_SEED);
    let hashes: Vec<u64> = (0..800u64)
        .map(|i| i.wrapping_mul(0x9E37_79B9_7F4A_7C15))
        .collect();
    for &h in &hashes {
        assert!(producer.insert(h));
    }
    let full = expect_full(producer.publish());
    let (consumer, meta) = assemble_full(&full);
    assert_eq!(consumer.len(), 800);

    for &h in &hashes[..600] {
        producer.remove(h);
    }
    let state = expect_full(producer.publish());
    assert!(state.epoch() > meta.filter_epoch);

    let (converged, _) = assemble_full(&state);
    assert_eq!(converged.len(), producer.len());
    assert_eq!(converged.len(), 200);
    for &h in &hashes[600..] {
        assert!(converged.contains(h));
    }
}

#[test]
fn churn_full_thresholds() {
    let nb = 1usize << 22;
    assert!(!churn_wants_full(nb / 3, nb));
    assert!(churn_wants_full(nb / 3 + 1, nb));

    let nb = 1usize << 24;
    let cap_dirty = MAX_DELTA_BYTES / DELTA_ENTRY_BYTES;
    assert!(cap_dirty + 1 < nb / 3);
    assert!(!churn_wants_full(cap_dirty, nb));
    assert!(churn_wants_full(cap_dirty + 1, nb));
}

#[test]
fn replacement_producer_resumes_epoch_chain() {
    let mut producer = SnapshotProducer::new(3, 64, DEFAULT_FILTER_SEED);
    assert!(producer.insert(1));
    let state = expect_full(producer.publish());
    let (mut consumer, meta) = assemble_full(&state);
    assert!(producer.insert(2));
    expect_delta(producer.publish());

    let mut replacement =
        SnapshotProducer::new_with_epoch(3, 64, producer.seed(), producer.epoch());
    for h in [1u64, 2, 3] {
        assert!(replacement.insert(h));
    }
    let state = expect_full(replacement.publish());
    assert!(state.epoch() > meta.filter_epoch);
    let (mut rebuilt, meta) = assemble_full(&state);
    assert_eq!(rebuilt.len(), 3);
    std::mem::swap(&mut consumer, &mut rebuilt);

    assert!(replacement.insert(4));
    let delta = expect_delta(replacement.publish());
    apply_delta(&mut consumer, meta.filter_epoch, &delta).unwrap();
    assert!(consumer.contains(4));
}

#[test]
fn snapshot_roundtrip_and_delta() {
    let seed = DEFAULT_FILTER_SEED;
    let mut producer = SnapshotProducer::new(42, 1000, seed);
    let hashes: Vec<u64> = (0..600u64)
        .map(|i| i.wrapping_mul(0x9E37_79B9_7F4A_7C15))
        .collect();
    for &h in &hashes {
        assert!(producer.insert(h));
    }
    let state = producer.full_snapshot();
    let (mut loaded, meta) = assemble_full(&state);
    assert_eq!(meta.dc_worker_id, 42);
    assert_eq!(meta.filter_epoch, 1);
    for &h in &hashes {
        assert!(loaded.contains(h));
    }

    for &h in &hashes[..60] {
        producer.remove(h);
    }
    for i in 600..650u64 {
        producer.insert(i.wrapping_mul(0x9E37_79B9_7F4A_7C15));
    }
    let delta = expect_delta(producer.publish());
    let info = apply_delta(&mut loaded, meta.filter_epoch, &delta).unwrap();
    assert_eq!(info.new_epoch, meta.filter_epoch + 1);
    assert_eq!(loaded.len(), 590);
    assert!(matches!(
        apply_delta(&mut loaded, 99, &delta),
        Err(DeltaError::BaseEpochGap {
            expected: 99,
            actual: 1
        })
    ));
}

#[test]
fn multi_chunk_assembly_matches_single_chunk() {
    let mut producer = SnapshotProducer::new(5, 4096, DEFAULT_FILTER_SEED);
    for i in 0..3000u64 {
        assert!(producer.insert(i.wrapping_mul(0x9E37_79B9_7F4A_7C15)));
    }
    let state = producer.full_snapshot();
    assert!(state.num_buckets() > 8);

    let (mono, mono_meta) = assemble_chunks(state.chunks_with(state.num_buckets())).unwrap();
    for per in [1usize, 7, 64, state.num_buckets() / 2] {
        let chunks: Vec<Vec<u8>> = state.chunks_with(per).collect();
        assert_eq!(chunks.len(), state.num_buckets().div_ceil(per));
        let (multi, meta) = assemble_chunks(&chunks).unwrap();
        assert_eq!(meta, mono_meta);
        assert_eq!(multi.to_raw_buckets(), mono.to_raw_buckets());
        assert_eq!(multi.len(), mono.len());
    }
}

#[test]
fn chunk_corruption_and_sequence_errors_are_rejected() {
    let mut producer = SnapshotProducer::new(1, 512, DEFAULT_FILTER_SEED);
    for i in 0..400u64 {
        assert!(producer.insert(i.wrapping_mul(0x9E37_79B9_7F4A_7C15)));
    }
    let state = producer.full_snapshot();
    let chunks: Vec<Vec<u8>> = state.chunks_with(state.num_buckets() / 4).collect();
    assert!(chunks.len() >= 3);

    let mut bad = chunks[1].clone();
    bad[SNAP_HEADER_LEN + CHUNK_BODY_PREFIX + 5] ^= 0xFF;
    let mut assembler = SnapshotAssembler::new();
    assert!(assembler.push(&chunks[0]).unwrap().is_none());
    assert!(matches!(
        assembler.push(&bad),
        Err(SnapshotError::ChunkChecksumMismatch)
    ));
    assert!(assembler.push(&chunks[1]).is_err());

    let mut assembler = SnapshotAssembler::new();
    assert!(assembler.push(&chunks[0]).unwrap().is_none());
    assert!(assembler.push(&chunks[2]).is_err());

    let mut assembler = SnapshotAssembler::new();
    assert!(assembler.push(&chunks[0]).unwrap().is_none());
    assert!(assembler.push(&chunks[1]).unwrap().is_none());
    assert!(assembler.push(&chunks[1]).is_err());

    let mut assembler = SnapshotAssembler::new();
    assert!(assembler.push(&chunks[2]).is_err());

    let mut assembler = SnapshotAssembler::new();
    assert!(assembler.push(&chunks[0]).unwrap().is_none());
    assert!(assembler.push(&chunks[2]).is_err());
    let mut completed = None;
    for chunk in &chunks {
        completed = assembler.push(chunk).unwrap();
    }
    let (filter, _) = completed.expect("complete");
    assert_eq!(filter.len(), producer.len());
}

#[test]
fn header_corruption_is_rejected() {
    let mut producer = SnapshotProducer::new(1, 64, DEFAULT_FILTER_SEED);
    assert!(producer.insert(7));
    let state = producer.full_snapshot();
    let chunk = state.chunks().next().unwrap();

    let mut bad_magic = chunk.clone();
    bad_magic[0] ^= 0xFF;
    assert!(parse_chunk(&bad_magic).is_err());

    let mut bad_version = chunk.clone();
    bad_version[4] ^= 0xFF;
    assert!(matches!(
        parse_chunk(&bad_version),
        Err(SnapshotError::VersionMismatch { .. })
    ));

    let mut bad_params = chunk.clone();
    bad_params[9] = 8;
    assert!(parse_chunk(&bad_params).is_err());

    let delta_on_chunk_parser = {
        let mut producer = SnapshotProducer::new(1, 64, DEFAULT_FILTER_SEED);
        producer.full_snapshot();
        producer.insert(1);
        expect_delta(producer.publish())
    };
    assert!(is_delta(&delta_on_chunk_parser));
    assert!(!is_chunk(&delta_on_chunk_parser));
    assert!(parse_chunk(&delta_on_chunk_parser).is_err());
    assert!(is_chunk(&chunk));
}

/// Force frequent kick chains (small capacity relative to insert count) so
/// `CuckooFilter`'s reused `kick_scratch` buffer is exercised on nearly every
/// insert, then converge a consumer through the full delta stream and check
/// it matches the producer byte-for-byte. A stale or mis-cleared scratch
/// buffer would corrupt the dirty-bucket reporting of kicked inserts, which
/// would either under- or over-report dirty buckets and desync the consumer.
#[test]
fn kicked_inserts_reuse_scratch_and_converge_byte_for_byte() {
    let mut producer = SnapshotProducer::new(1, 64, DEFAULT_FILTER_SEED);
    let full = expect_full(producer.publish());
    let (mut consumer, mut epoch) = {
        let (filter, meta) = assemble_full(&full);
        (filter, meta.filter_epoch)
    };

    let mut inserted = Vec::new();
    for i in 0..300u64 {
        let h = spread(i);
        if !producer.insert(h) {
            break;
        }
        inserted.push(h);
        match producer.publish() {
            Publish::Delta(delta) => {
                let info = apply_delta(&mut consumer, epoch, &delta).unwrap();
                epoch = info.new_epoch;
            }
            Publish::Full(state) => {
                let (filter, meta) = assemble_full(&state);
                consumer = filter;
                epoch = meta.filter_epoch;
            }
            Publish::Unchanged => {}
        }
    }

    assert!(
        inserted.len() > 50,
        "test should exercise many kicked inserts, only got {}",
        inserted.len()
    );
    for &h in &inserted {
        assert!(consumer.contains(h));
    }
    assert_eq!(consumer.len(), inserted.len());
    assert_eq!(consumer.len(), producer.len());

    let (from_producer, _) = assemble_full(&producer.current_snapshot());
    assert_eq!(consumer.to_raw_buckets(), from_producer.to_raw_buckets());
}

/// A bucket touched twice in one publish window (removed, then re-inserted)
/// must not cost any wire bytes even though it was marked dirty, while a
/// genuinely new bucket in the same window still ships. This is exactly the
/// comparison `build_delta_for_buckets` makes against the `BucketPages`-backed
/// `last_shipped` baseline (not just the all-or-nothing `Unchanged` case).
#[test]
fn partial_cancellation_excludes_only_reverted_buckets() {
    let mut producer = SnapshotProducer::new(4, 4096, DEFAULT_FILTER_SEED);
    let real_a = spread(1);
    let cancel = spread(2);
    let real_b = spread(3);
    for &h in &[real_a, cancel] {
        assert!(producer.insert(h));
    }
    let full = expect_full(producer.publish());
    let (mut consumer, meta) = assemble_full(&full);
    assert_eq!(consumer.len(), 2);

    producer.remove(cancel);
    assert!(producer.insert(cancel));
    assert!(producer.insert(real_b));

    let delta = expect_delta(producer.publish());
    assert_eq!(
        delta_bucket_count(&delta),
        1,
        "the cancelled bucket must be filtered out, only real_b's bucket should ship"
    );

    apply_delta(&mut consumer, meta.filter_epoch, &delta).unwrap();
    assert_eq!(consumer.len(), 3);
    assert!(consumer.contains(real_a));
    assert!(consumer.contains(cancel));
    assert!(consumer.contains(real_b));
}

/// A `SnapshotState` handed to a slow subscriber shares pages with
/// `filter.buckets` at capture time. `last_shipped` re-baselines from the same
/// `filter.buckets` on every `full_snapshot()`, so this checks the two uses of
/// that clone never alias: churn after a second rebaseline must not perturb
/// either earlier point-in-time snapshot.
#[test]
fn retained_snapshot_survives_full_snapshot_rebaseline() {
    let mut producer = SnapshotProducer::new(6, 4096, DEFAULT_FILTER_SEED);
    for h in 0..500u64 {
        assert!(producer.insert(spread(h)));
    }
    let retained = producer.current_snapshot();
    let (retained_filter, _) = assemble_full(&retained);
    assert_eq!(retained_filter.len(), 500);

    for h in 500..900u64 {
        assert!(producer.insert(spread(h)));
    }
    for h in 0..300u64 {
        producer.remove(spread(h));
    }
    // Exercise `last_shipped = filter.buckets.clone()` a second time, on top
    // of a filter that has already diverged from the first snapshot via COW.
    let second_state = producer.full_snapshot();
    let (second_filter, _) = assemble_full(&second_state);
    assert_eq!(second_filter.len(), 900 - 300);

    for h in 900..1200u64 {
        assert!(producer.insert(spread(h)));
    }
    let _ = producer.publish();

    let (retained_after, _) = assemble_full(&retained);
    assert_eq!(
        retained_after.to_raw_buckets(),
        retained_filter.to_raw_buckets(),
        "later churn must not perturb the first retained snapshot"
    );

    let (second_after, _) = assemble_full(&second_state);
    assert_eq!(second_after.len(), 900 - 300);
    for h in 300..900u64 {
        assert!(second_after.contains(spread(h)));
    }
    for h in 0..300u64 {
        assert!(!second_after.contains(spread(h)));
    }
    for h in 900..1200u64 {
        assert!(
            !second_after.contains(spread(h)),
            "later churn must not leak into the second retained snapshot"
        );
    }
}

#[test]
fn probes_match_direct_contains() {
    let seed = DEFAULT_FILTER_SEED;
    let mut filter = CuckooFilter::with_capacity_seeded(256, seed);
    let chain: Vec<u64> = (0..64u64)
        .map(|i| i.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0x1234)
        .collect();
    for &h in &chain[..40] {
        assert!(filter.insert(h));
    }
    let probes = probes_for(&chain, seed);
    assert_eq!(filter_overlap(&filter, &probes), 40);
    for &h in &chain[..40] {
        assert!(filter.contains(h));
    }
}
