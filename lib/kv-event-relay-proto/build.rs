// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("cargo:rerun-if-changed=proto/relay.proto");
    let out_dir = std::env::var("OUT_DIR")?;
    // Emit a descriptor set so the relay can expose reflection without a
    // checked-out proto tree.
    let descriptor_path = std::path::PathBuf::from(out_dir).join("relay_descriptor.bin");
    tonic_prost_build::configure()
        .build_server(true)
        .build_client(true)
        // Keep payloads reference-counted so replay and fan-out do not deep-copy
        // every frame.
        .bytes(".")
        .file_descriptor_set_path(&descriptor_path)
        .compile_protos(&["proto/relay.proto"], &["proto"])?;
    Ok(())
}
