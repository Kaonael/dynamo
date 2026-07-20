// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! CBI1 — the wire encoding for Cuckoo Bucket Images: how a Relay ships the
//! upstream CKF publication stream (absolute bucket images plus full-lane
//! snapshots) to the Global Gateway.
//!
//! Deltas carry the images of one publication verbatim. Full snapshots carry
//! the flat lane word array in bounded chunks (a dense dump is smaller than a
//! sparse image list at production load factors, and chunking keeps every
//! frame under gRPC message limits). Every frame repeats the filter format
//! identity so the consumer can gate drift per frame, carries the publisher
//! epoch for continuity checks, and checksums its body; session identity,
//! heartbeats, and resubscription stay in the gRPC envelope exactly as
//! before.

use xxhash_rust::xxh3;

/// Frame magic.
pub const IMAGES_MAGIC: [u8; 4] = *b"CBI1";
/// Wire-format version of this module.
pub const IMAGES_WIRE_VERSION: u16 = 1;
/// Header length in bytes.
pub const IMAGES_HEADER_LEN: usize = 48;
/// Snapshot chunk payload target: buckets per chunk (4 MiB of lane words).
pub const SNAPSHOT_CHUNK_BUCKETS: usize = 512 * 1024;

/// Largest single CBI1 frame a producer will ever emit — a full snapshot chunk
/// with a complete `SNAPSHOT_CHUNK_BUCKETS` payload. A delta is capped to this
/// too (see [`max_delta_images`]); above it the producer falls back to a
/// chunked snapshot. The gRPC message limit must clear this plus its envelope.
pub const IMAGES_MAX_FRAME_BYTES: usize = IMAGES_HEADER_LEN + 16 + SNAPSHOT_CHUNK_BUCKETS * 8;

/// Bytes of delta body per image (`bucket: u32` + `value: u64`).
const DELTA_IMAGE_BYTES: usize = 12;
/// Fixed delta body prefix (`base_epoch: u64` + `image count: u32`).
const DELTA_BODY_PREFIX: usize = 12;

/// Most images one delta frame may carry while staying within
/// [`IMAGES_MAX_FRAME_BYTES`]. A publication touching more buckets than this is
/// shipped as a chunked snapshot instead.
pub const fn max_delta_images() -> usize {
    (IMAGES_MAX_FRAME_BYTES - IMAGES_HEADER_LEN - DELTA_BODY_PREFIX) / DELTA_IMAGE_BYTES
}

const FLAG_SNAPSHOT_CHUNK: u16 = 1;
const FLAG_DELTA: u16 = 2;

/// Upstream CKF format version carried by CBI1.
pub const FORMAT_VERSION: u16 = 1;
/// Upstream fingerprint width; one bucket word packs four fingerprints.
pub const FINGERPRINT_BITS: u8 = 16;
/// Upstream slots per bucket; one absolute bucket image is one `u64`.
pub const SLOTS_PER_BUCKET: u8 = 4;

/// The upstream format identity required to decode and address bucket images.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FilterFormat {
    pub seed: u64,
    pub bucket_count: usize,
}

/// Why a received filter format cannot be consumed.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum FormatError {
    #[error("unsupported CKF format version {0} (expected {FORMAT_VERSION})")]
    Version(u16),
    #[error("unsupported fingerprint width {0} (expected {FINGERPRINT_BITS})")]
    FingerprintBits(u8),
    #[error("unsupported slots per bucket {0} (expected {SLOTS_PER_BUCKET})")]
    SlotsPerBucket(u8),
    #[error("bucket count {0} is not a power of two >= 2")]
    BucketCount(usize),
    #[error("format mismatch: expected {expected:?}, received seed {seed:#x} buckets {buckets}")]
    Mismatch {
        expected: FilterFormat,
        seed: u64,
        buckets: usize,
    },
}

impl FilterFormat {
    pub fn new(seed: u64, bucket_count: usize) -> Result<Self, FormatError> {
        if !bucket_count.is_power_of_two() || bucket_count < 2 {
            return Err(FormatError::BucketCount(bucket_count));
        }
        Ok(Self { seed, bucket_count })
    }

    pub fn validate(
        &self,
        version: u16,
        seed: u64,
        bucket_count: usize,
        fingerprint_bits: u8,
        slots_per_bucket: u8,
    ) -> Result<(), FormatError> {
        if version != FORMAT_VERSION {
            return Err(FormatError::Version(version));
        }
        if fingerprint_bits != FINGERPRINT_BITS {
            return Err(FormatError::FingerprintBits(fingerprint_bits));
        }
        if slots_per_bucket != SLOTS_PER_BUCKET {
            return Err(FormatError::SlotsPerBucket(slots_per_bucket));
        }
        if seed != self.seed || bucket_count != self.bucket_count {
            return Err(FormatError::Mismatch {
                expected: *self,
                seed,
                buckets: bucket_count,
            });
        }
        Ok(())
    }
}

/// One absolute bucket word addressed within a filter lane.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BucketImage {
    pub bucket: u32,
    pub value: u64,
}

/// Why a received frame cannot be consumed.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum ImagesWireError {
    #[error("frame shorter than the CBI1 header")]
    Truncated,
    #[error("bad magic")]
    Magic,
    #[error("unsupported CBI1 wire version {0}")]
    WireVersion(u16),
    #[error("unknown frame flags {0:#06x}")]
    Flags(u16),
    #[error("body checksum mismatch")]
    Checksum,
    #[error("frame body is malformed")]
    Malformed,
    #[error(transparent)]
    Format(#[from] FormatError),
    #[error("delta base epoch {received} does not extend current epoch {current}")]
    EpochGap { current: u64, received: u64 },
    #[error("snapshot chunk sequence violation")]
    ChunkSequence,
    #[error("snapshot chunks do not cover the lane")]
    IncompleteCoverage,
}

/// Decoded frame header, identical for both frame kinds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ImagesHeader {
    pub dc_worker_id: u64,
    /// Publisher epoch this frame produces (snapshot) or advances to (delta).
    pub epoch: u64,
    pub seed: u64,
    pub bucket_count: u64,
}

/// One decoded frame.
#[derive(Debug, PartialEq, Eq)]
pub enum ImagesFrame {
    SnapshotChunk {
        header: ImagesHeader,
        chunk_index: u32,
        chunk_count: u32,
        bucket_offset: u64,
        words: Vec<u64>,
    },
    Delta {
        header: ImagesHeader,
        base_epoch: u64,
        images: Vec<BucketImage>,
    },
}

impl ImagesFrame {
    pub fn header(&self) -> ImagesHeader {
        match self {
            ImagesFrame::SnapshotChunk { header, .. } | ImagesFrame::Delta { header, .. } => {
                *header
            }
        }
    }
}

fn write_header(out: &mut Vec<u8>, flags: u16, format: FilterFormat, dc: u64, epoch: u64) {
    out.extend_from_slice(&IMAGES_MAGIC);
    out.extend_from_slice(&IMAGES_WIRE_VERSION.to_le_bytes());
    out.extend_from_slice(&flags.to_le_bytes());
    out.push(FINGERPRINT_BITS);
    out.push(SLOTS_PER_BUCKET);
    // CKF format version at offset 10..12: so a future upstream layout
    // bump is detectable on the wire instead of decoding under the wrong
    // format. Distinct from IMAGES_WIRE_VERSION (the frame encoding version).
    out.extend_from_slice(&FORMAT_VERSION.to_le_bytes());
    out.extend_from_slice(&format.seed.to_le_bytes());
    out.extend_from_slice(&(format.bucket_count as u64).to_le_bytes());
    out.extend_from_slice(&dc.to_le_bytes());
    out.extend_from_slice(&epoch.to_le_bytes());
    // Checksum placeholder; patched once the body is written.
    out.extend_from_slice(&[0u8; 4]);
}

fn patch_checksum(frame: &mut [u8]) {
    let checksum = xxh3::xxh3_64(&frame[IMAGES_HEADER_LEN..]) as u32;
    frame[IMAGES_HEADER_LEN - 4..IMAGES_HEADER_LEN].copy_from_slice(&checksum.to_le_bytes());
}

/// Encode one publication's images as a delta frame advancing
/// `base_epoch -> epoch`.
pub fn encode_delta(
    format: FilterFormat,
    dc: u64,
    base_epoch: u64,
    epoch: u64,
    images: &[BucketImage],
) -> Vec<u8> {
    let mut out = Vec::with_capacity(IMAGES_HEADER_LEN + 12 + images.len() * 12);
    write_header(&mut out, FLAG_DELTA, format, dc, epoch);
    out.extend_from_slice(&base_epoch.to_le_bytes());
    out.extend_from_slice(&(images.len() as u32).to_le_bytes());
    for image in images {
        out.extend_from_slice(&image.bucket.to_le_bytes());
        out.extend_from_slice(&image.value.to_le_bytes());
    }
    patch_checksum(&mut out);
    out
}

/// Encode a full-lane snapshot as bounded chunks establishing `epoch`.
pub fn encode_snapshot_chunks(
    format: FilterFormat,
    dc: u64,
    epoch: u64,
    words: &[u64],
) -> Vec<Vec<u8>> {
    // `FilterFormat` guarantees bucket_count >= 2, so at least one chunk is
    // always produced.
    assert_eq!(words.len(), format.bucket_count);
    let chunk_count = words.len().div_ceil(SNAPSHOT_CHUNK_BUCKETS) as u32;
    let mut frames = Vec::with_capacity(chunk_count as usize);
    for (chunk_index, chunk) in words.chunks(SNAPSHOT_CHUNK_BUCKETS).enumerate() {
        let mut out = Vec::with_capacity(IMAGES_HEADER_LEN + 16 + chunk.len() * 8);
        write_header(&mut out, FLAG_SNAPSHOT_CHUNK, format, dc, epoch);
        out.extend_from_slice(&(chunk_index as u32).to_le_bytes());
        out.extend_from_slice(&chunk_count.to_le_bytes());
        out.extend_from_slice(&((chunk_index * SNAPSHOT_CHUNK_BUCKETS) as u64).to_le_bytes());
        for &word in chunk {
            out.extend_from_slice(&word.to_le_bytes());
        }
        patch_checksum(&mut out);
        frames.push(out);
    }
    frames
}

fn read_u16(bytes: &[u8], at: usize) -> u16 {
    u16::from_le_bytes(bytes[at..at + 2].try_into().expect("bounds checked"))
}

fn read_u32(bytes: &[u8], at: usize) -> u32 {
    u32::from_le_bytes(bytes[at..at + 4].try_into().expect("bounds checked"))
}

fn read_u64(bytes: &[u8], at: usize) -> u64 {
    u64::from_le_bytes(bytes[at..at + 8].try_into().expect("bounds checked"))
}

/// Read the filter format identity a frame advertises, without validating the
/// body. Lets a consumer that has not yet adopted a format bootstrap it from
/// the first frame (subsequent frames are then gated by [`decode`]).
pub fn peek_format(bytes: &[u8]) -> Result<FilterFormat, ImagesWireError> {
    if bytes.len() < IMAGES_HEADER_LEN {
        return Err(ImagesWireError::Truncated);
    }
    if bytes[0..4] != IMAGES_MAGIC {
        return Err(ImagesWireError::Magic);
    }
    let wire_version = read_u16(bytes, 4);
    if wire_version != IMAGES_WIRE_VERSION {
        return Err(ImagesWireError::WireVersion(wire_version));
    }
    let seed = read_u64(bytes, 12);
    let bucket_count =
        usize::try_from(read_u64(bytes, 20)).map_err(|_| ImagesWireError::Malformed)?;
    FilterFormat::new(seed, bucket_count).map_err(ImagesWireError::from)
}

/// Decode and integrity-check one frame against the consumer's expected
/// format.
pub fn decode(expected: FilterFormat, bytes: &[u8]) -> Result<ImagesFrame, ImagesWireError> {
    if bytes.len() < IMAGES_HEADER_LEN {
        return Err(ImagesWireError::Truncated);
    }
    if bytes[0..4] != IMAGES_MAGIC {
        return Err(ImagesWireError::Magic);
    }
    let wire_version = read_u16(bytes, 4);
    if wire_version != IMAGES_WIRE_VERSION {
        return Err(ImagesWireError::WireVersion(wire_version));
    }
    let flags = read_u16(bytes, 6);
    let fingerprint_bits = bytes[8];
    let slots_per_bucket = bytes[9];
    let format_version = read_u16(bytes, 10);
    let seed = read_u64(bytes, 12);
    let bucket_count = read_u64(bytes, 20);
    let dc_worker_id = read_u64(bytes, 28);
    let epoch = read_u64(bytes, 36);
    let checksum = read_u32(bytes, IMAGES_HEADER_LEN - 4);
    let body = &bytes[IMAGES_HEADER_LEN..];
    if xxh3::xxh3_64(body) as u32 != checksum {
        return Err(ImagesWireError::Checksum);
    }
    expected.validate(
        format_version,
        seed,
        usize::try_from(bucket_count).map_err(|_| ImagesWireError::Malformed)?,
        fingerprint_bits,
        slots_per_bucket,
    )?;
    let header = ImagesHeader {
        dc_worker_id,
        epoch,
        seed,
        bucket_count,
    };

    match flags {
        FLAG_DELTA => {
            if body.len() < 12 {
                return Err(ImagesWireError::Malformed);
            }
            let base_epoch = read_u64(body, 0);
            let count = read_u32(body, 8) as usize;
            let expected_len = 12 + count * 12;
            if body.len() != expected_len {
                return Err(ImagesWireError::Malformed);
            }
            let mut images = Vec::with_capacity(count);
            for index in 0..count {
                let at = 12 + index * 12;
                let bucket = read_u32(body, at);
                let value = read_u64(body, at + 4);
                if u64::from(bucket) >= bucket_count {
                    return Err(ImagesWireError::Malformed);
                }
                images.push(BucketImage { bucket, value });
            }
            Ok(ImagesFrame::Delta {
                header,
                base_epoch,
                images,
            })
        }
        FLAG_SNAPSHOT_CHUNK => {
            if body.len() < 16 {
                return Err(ImagesWireError::Malformed);
            }
            let chunk_index = read_u32(body, 0);
            let chunk_count = read_u32(body, 4);
            let bucket_offset = read_u64(body, 8);
            let words_bytes = &body[16..];
            if !words_bytes.len().is_multiple_of(8) {
                return Err(ImagesWireError::Malformed);
            }
            let words: Vec<u64> = words_bytes
                .chunks_exact(8)
                .map(|chunk| u64::from_le_bytes(chunk.try_into().expect("chunked by 8")))
                .collect();
            let end_bucket = bucket_offset
                .checked_add(words.len() as u64)
                .ok_or(ImagesWireError::Malformed)?;
            if end_bucket > bucket_count {
                return Err(ImagesWireError::Malformed);
            }
            Ok(ImagesFrame::SnapshotChunk {
                header,
                chunk_index,
                chunk_count,
                bucket_offset,
                words,
            })
        }
        other => Err(ImagesWireError::Flags(other)),
    }
}

// ---------------------------------------------------------------------------
// Consumer-side snapshot assembly
// ---------------------------------------------------------------------------

/// Accumulates one DC's snapshot chunks in order into a sparse image list.
pub struct SnapshotAssembly {
    epoch: u64,
    chunk_count: u32,
    next_chunk: u32,
    next_bucket: u64,
    bucket_count: u64,
    images: Vec<BucketImage>,
}

impl SnapshotAssembly {
    pub fn new(format: FilterFormat) -> Self {
        Self {
            epoch: 0,
            chunk_count: 0,
            next_chunk: 0,
            next_bucket: 0,
            bucket_count: format.bucket_count as u64,
            images: Vec::new(),
        }
    }

    pub fn reset(&mut self) {
        self.epoch = 0;
        self.chunk_count = 0;
        self.next_chunk = 0;
        self.next_bucket = 0;
        self.images.clear();
    }

    /// Absorb one chunk; returns the completed sparse dump (nonzero buckets)
    /// with its epoch when the final chunk lands.
    pub fn absorb(
        &mut self,
        frame: &ImagesFrame,
    ) -> Result<Option<(u64, Vec<BucketImage>)>, ImagesWireError> {
        let ImagesFrame::SnapshotChunk {
            header,
            chunk_index,
            chunk_count,
            bucket_offset,
            words,
        } = frame
        else {
            return Err(ImagesWireError::ChunkSequence);
        };
        if *chunk_index == 0 {
            self.reset();
            self.epoch = header.epoch;
            self.chunk_count = *chunk_count;
        }
        if *chunk_index != self.next_chunk
            || *chunk_count != self.chunk_count
            || header.epoch != self.epoch
            || *bucket_offset != self.next_bucket
        {
            self.reset();
            return Err(ImagesWireError::ChunkSequence);
        }
        for (offset, &word) in words.iter().enumerate() {
            if word != 0 {
                self.images.push(BucketImage {
                    bucket: (*bucket_offset + offset as u64) as u32,
                    value: word,
                });
            }
        }
        self.next_bucket += words.len() as u64;
        self.next_chunk += 1;
        if self.next_chunk == self.chunk_count {
            if self.next_bucket != self.bucket_count {
                self.reset();
                return Err(ImagesWireError::IncompleteCoverage);
            }
            let epoch = self.epoch;
            let images = std::mem::take(&mut self.images);
            self.reset();
            return Ok(Some((epoch, images)));
        }
        Ok(None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn format() -> FilterFormat {
        FilterFormat::new(0x5EED, 1 << 10).unwrap()
    }

    #[test]
    fn delta_round_trips() {
        let images = vec![
            BucketImage {
                bucket: 3,
                value: 0xAAAA_BBBB_CCCC_DDDD,
            },
            BucketImage {
                bucket: 1000,
                value: 42,
            },
        ];
        let bytes = encode_delta(format(), 7, 4, 5, &images);
        let frame = decode(format(), &bytes).unwrap();
        match frame {
            ImagesFrame::Delta {
                header,
                base_epoch,
                images: decoded,
            } => {
                assert_eq!(header.dc_worker_id, 7);
                assert_eq!(header.epoch, 5);
                assert_eq!(base_epoch, 4);
                assert_eq!(decoded, images);
            }
            other => panic!("expected delta, got {other:?}"),
        }
    }

    #[test]
    fn corruption_and_drift_are_rejected() {
        let images = vec![BucketImage {
            bucket: 1,
            value: 2,
        }];
        let mut bytes = encode_delta(format(), 7, 0, 1, &images);
        *bytes.last_mut().unwrap() ^= 1;
        assert_eq!(
            decode(format(), &bytes),
            Err(ImagesWireError::Checksum),
            "body corruption must fail the checksum"
        );

        let bytes = encode_delta(format(), 7, 0, 1, &images);
        let other = FilterFormat::new(0x5EED ^ 1, 1 << 10).unwrap();
        assert!(
            matches!(decode(other, &bytes), Err(ImagesWireError::Format(_))),
            "seed drift must be gated"
        );

        let mut bytes = encode_delta(format(), 7, 0, 1, &images);
        bytes[0] = b'X';
        assert_eq!(decode(format(), &bytes), Err(ImagesWireError::Magic));
    }

    #[test]
    fn wire_format_version_is_carried_and_validated() {
        // The encoder stamps FORMAT_VERSION at header offset 10..12, and
        // decode validates the on-wire value (not a compile-time constant), so
        // a future CKF layout bump is detectable. The checksum covers only the
        // body, so corrupting the header version exercises the version gate.
        let images = vec![BucketImage {
            bucket: 1,
            value: 2,
        }];
        let good = encode_delta(format(), 7, 0, 1, &images);
        assert_eq!(read_u16(&good, 10), FORMAT_VERSION, "version is on the wire");
        assert!(decode(format(), &good).is_ok());

        let mut bad = good.clone();
        bad[10] = bad[10].wrapping_add(1);
        assert!(
            matches!(
                decode(format(), &bad),
                Err(ImagesWireError::Format(FormatError::Version(_)))
            ),
            "on-wire format_version drift must be rejected"
        );
    }

    #[test]
    fn snapshot_chunks_assemble_into_sparse_dump() {
        let format = format();
        let mut words = vec![0u64; format.bucket_count];
        words[0] = 11;
        words[513] = 22;
        words[format.bucket_count - 1] = 33;
        // Force multiple chunks with a tiny chunk size by encoding manually
        // through the public encoder (chunk size is fixed; the lane fits in
        // one chunk here, so also test the single-chunk path).
        let frames = encode_snapshot_chunks(format, 9, 6, &words);
        let mut assembly = SnapshotAssembly::new(format);
        let mut completed = None;
        for bytes in &frames {
            let frame = decode(format, bytes).unwrap();
            if let Some(done) = assembly.absorb(&frame).unwrap() {
                completed = Some(done);
            }
        }
        let (epoch, images) = completed.expect("assembly completes");
        assert_eq!(epoch, 6);
        assert_eq!(
            images,
            vec![
                BucketImage {
                    bucket: 0,
                    value: 11
                },
                BucketImage {
                    bucket: 513,
                    value: 22
                },
                BucketImage {
                    bucket: (format.bucket_count - 1) as u32,
                    value: 33
                },
            ]
        );
    }

    #[test]
    fn deltas_and_snapshot_assembly_produce_identical_words() {
        let format = format();
        let mut expected = vec![0u64; format.bucket_count];
        let mut via_deltas = vec![0u64; format.bucket_count];

        // Build some content through images derived from a scratch replica of
        // one lane (values are arbitrary nonzero words for wire purposes).
        let deltas: Vec<Vec<BucketImage>> = (0..5)
            .map(|round| {
                (0..8)
                    .map(|slot| BucketImage {
                        bucket: (round * 97 + slot * 13) % format.bucket_count as u32,
                        value: 0x1111_0000_0000_0000 | u64::from(round * 8 + slot + 1),
                    })
                    .collect()
            })
            .collect();

        let mut epoch = 0u64;
        for images in &deltas {
            for image in images {
                expected[image.bucket as usize] = image.value;
            }
            let bytes = encode_delta(format, 1, epoch, epoch + 1, images);
            epoch += 1;
            match decode(format, &bytes).unwrap() {
                ImagesFrame::Delta { images, .. } => {
                    for image in images {
                        via_deltas[image.bucket as usize] = image.value;
                    }
                }
                other => panic!("expected delta, got {other:?}"),
            }
        }
        assert_eq!(via_deltas, expected);

        let frames = encode_snapshot_chunks(format, 1, epoch, &expected);
        let mut assembly = SnapshotAssembly::new(format);
        let mut via_snapshot = vec![0u64; format.bucket_count];
        for bytes in &frames {
            let frame = decode(format, bytes).unwrap();
            if let Some((_, images)) = assembly.absorb(&frame).unwrap() {
                for image in images {
                    via_snapshot[image.bucket as usize] = image.value;
                }
            }
        }
        assert_eq!(via_snapshot, expected);
    }
}
