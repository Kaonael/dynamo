// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! P2P model config file endpoint.
//!
//! Workers register a `model-config` endpoint that serves config files (config.json,
//! tokenizer.json, etc.) to frontends over the request plane. This allows frontends to
//! obtain model files directly from any available worker, without requiring HuggingFace access.

use std::path::{Path, PathBuf};

use anyhow::Context as _;
use dynamo_runtime::engine::AsyncEngineContextProvider;
use dynamo_runtime::pipeline::{
    AsyncEngine, Error, ManyOut, ResponseStream, SingleIn, async_trait,
};
use dynamo_runtime::protocols::annotated::Annotated;
use serde::{Deserialize, Serialize};

use crate::model_card::ModelDeploymentCard;

/// Name of the endpoint registered by workers for serving config files.
pub const MODEL_CONFIG_ENDPOINT: &str = "model-config";

/// Request for a single config file from a worker.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct ModelConfigRequest {
    pub filename: String,
}

/// Response containing file content as text.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct ModelConfigResponse(pub String);

/// Engine that serves model config files from a local directory.
pub struct ModelConfigEngine {
    model_dir: PathBuf,
}

impl ModelConfigEngine {
    pub fn new(model_dir: PathBuf) -> Self {
        Self { model_dir }
    }
}

#[async_trait]
impl AsyncEngine<SingleIn<ModelConfigRequest>, ManyOut<Annotated<ModelConfigResponse>>, Error>
    for ModelConfigEngine
{
    async fn generate(
        &self,
        input: SingleIn<ModelConfigRequest>,
    ) -> anyhow::Result<ManyOut<Annotated<ModelConfigResponse>>> {
        let (req, ctx) = input.into_parts();

        // Sanitize: only allow plain filenames, reject paths with separators or traversal
        let fname = Path::new(&req.filename)
            .file_name()
            .and_then(|f| f.to_str())
            .ok_or_else(|| anyhow::anyhow!("Invalid filename: {}", req.filename))?
            .to_string();

        let filepath = self.model_dir.join(&fname);
        let data = tokio::fs::read_to_string(&filepath)
            .await
            .with_context(|| format!("Config file not available: {fname}"))?;

        let stream =
            futures::stream::once(async move { Annotated::from_data(ModelConfigResponse(data)) });
        Ok(ResponseStream::new(Box::pin(stream), ctx.context()))
    }
}

/// Trait for P2P config file downloading.
#[async_trait]
pub trait P2pConfigDownloader: Send + Sync {
    async fn download(&self, card: &mut ModelDeploymentCard) -> anyhow::Result<()>;
}

/// Cache directory for files downloaded via P2P from workers.
/// Uses slugify_unique (with hash suffix) to avoid collisions between
/// models with similar source paths (e.g. "Org/Model-V1" vs "org/model-v1").
pub fn p2p_cache_dir(source_path: &str) -> PathBuf {
    let cache_root = crate::hub::get_model_express_cache_dir();
    let slug = dynamo_runtime::slug::Slug::slugify_unique(source_path);
    cache_root.join(format!("models--{slug}")).join("p2p")
}

#[cfg(test)]
mod tests {
    use super::*;
    use dynamo_runtime::pipeline::Context;
    use futures::StreamExt;
    use std::io::Write;
    use tempfile::tempdir;

    fn make_request(filename: &str) -> SingleIn<ModelConfigRequest> {
        Context::new(ModelConfigRequest {
            filename: filename.to_string(),
        })
    }

    #[tokio::test]
    async fn test_engine_serves_existing_file() {
        let dir = tempdir().unwrap();
        let content = r#"{"model_type": "llama"}"#;
        std::fs::write(dir.path().join("config.json"), content).unwrap();

        let engine = ModelConfigEngine::new(dir.path().to_path_buf());
        let result = engine.generate(make_request("config.json")).await;
        assert!(result.is_ok());

        let mut stream = result.unwrap();
        let item = stream.next().await.unwrap();
        assert_eq!(item.data.unwrap().0, content);
    }

    #[tokio::test]
    async fn test_engine_file_not_found() {
        let dir = tempdir().unwrap();
        let engine = ModelConfigEngine::new(dir.path().to_path_buf());
        let result = engine.generate(make_request("nonexistent.json")).await;
        assert!(result.is_err());
        assert!(
            result.unwrap_err().to_string().contains("Config file not available")
        );
    }

    #[tokio::test]
    async fn test_engine_rejects_path_traversal() {
        let dir = tempdir().unwrap();
        let engine = ModelConfigEngine::new(dir.path().to_path_buf());
        let result = engine.generate(make_request("../../../etc/passwd")).await;
        // Path::file_name() on "../../../etc/passwd" returns Some("passwd"),
        // so this becomes a "file not found" rather than "invalid filename"
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_engine_sanitizes_to_filename_only() {
        let dir = tempdir().unwrap();
        // Create "passwd" in the temp dir to prove sanitization strips the path
        std::fs::write(dir.path().join("passwd"), "safe content").unwrap();

        let engine = ModelConfigEngine::new(dir.path().to_path_buf());
        // "../etc/passwd" gets sanitized to just "passwd" via Path::file_name()
        let result = engine.generate(make_request("../etc/passwd")).await;
        assert!(result.is_ok());

        let mut stream = result.unwrap();
        let item = stream.next().await.unwrap();
        assert_eq!(item.data.unwrap().0, "safe content");
    }

    #[test]
    fn test_request_serde_roundtrip() {
        let req = ModelConfigRequest {
            filename: "tokenizer.json".to_string(),
        };
        let json = serde_json::to_string(&req).unwrap();
        let req2: ModelConfigRequest = serde_json::from_str(&json).unwrap();
        assert_eq!(req.filename, req2.filename);
    }

    #[test]
    fn test_response_serde_no_bloat() {
        let content = r#"{"key": "value"}"#;
        let resp = ModelConfigResponse(content.to_string());
        let json = serde_json::to_string(&resp).unwrap();
        // String serialization should be roughly 1:1, not array-of-numbers
        assert!(json.len() < content.len() * 2);
        let resp2: ModelConfigResponse = serde_json::from_str(&json).unwrap();
        assert_eq!(resp2.0, content);
    }

    #[test]
    fn test_p2p_cache_dir_format() {
        let dir = p2p_cache_dir("meta-llama/Meta-Llama-3-8B");
        let dir_str = dir.to_string_lossy();
        assert!(dir_str.contains("models--"));
        assert!(dir_str.ends_with("/p2p"));
    }

    #[test]
    fn test_p2p_cache_dir_unique_slugs() {
        let dir1 = p2p_cache_dir("Org/Model-V1");
        let dir2 = p2p_cache_dir("org/model-v1");
        // slugify_unique adds hash — different inputs should produce different dirs
        // (even if slug text looks similar)
        assert_ne!(dir1, dir2);
    }
}
