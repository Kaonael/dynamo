---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Frontend
---

The Dynamo Frontend is the API gateway for serving LLM inference requests. It provides OpenAI-compatible HTTP endpoints and KServe gRPC endpoints, handling request preprocessing, routing, and response formatting.

## Feature Matrix

| Feature | Status |
|---------|--------|
| OpenAI Chat Completions API (`/v1/chat/completions`) | ✅ Supported |
| OpenAI Completions API (`/v1/completions`) | ✅ Supported |
| OpenAI Embeddings API (`/v1/embeddings`) | ✅ Supported |
| OpenAI Responses API (`/v1/responses`) | ✅ Supported |
| OpenAI Models API (`/v1/models`) | ✅ Supported |
| Image Generation (`/v1/images/generations`) | ✅ Supported |
| Video Generation (`/v1/videos/generations`) | ✅ Supported |
| Anthropic Messages API (`/v1/messages`) | 🧪 Experimental |
| KServe gRPC v2 API | ✅ Supported |
| Streaming responses (SSE) | ✅ Supported |
| Multi-model serving | ✅ Supported |
| Integrated KV-aware routing | ✅ Supported |
| Tool calling | ✅ Supported |
| TLS (HTTPS) | ✅ Supported |
| Swagger UI (`/docs`) | ✅ Supported |
| NVIDIA request extensions (`nvext`) | ✅ Supported |

## Quick Start

### Prerequisites

- Dynamo platform installed
- `etcd` and `nats-server -js` running
- At least one backend worker registered

### HTTP Frontend

```bash
python -m dynamo.frontend --http-port 8000
```

This starts an OpenAI-compatible HTTP server with integrated pre/post processing and routing. Backends are auto-discovered when they call `register_model`.

The frontend does the pre and post processing. To do this it will need access to the model configuration files: `config.json`, `tokenizer.json`, `tokenizer_config.json`, etc. It does not need the weights.

The frontend automatically obtains these files using the following priority chain:

1. **Local HF cache** — if files are already cached locally, no network calls are made.
2. **ModelExpress server** — downloads from [modelexpress-server](https://github.com/ai-dynamo/modelexpress) if configured. We recommend setting up ModelExpress with a shared folder (e.g. a Kubernetes PVC) so the model is only downloaded once across the cluster.
3. **P2P from workers** — the frontend downloads config files directly from any available worker over the request plane (HTTP, TCP, or NATS). Workers automatically register a `model-config` endpoint alongside their inference endpoints. This requires no additional setup and works with any transport mode.
4. **Direct HuggingFace download** — last resort fallback.

For **private or customized models** that are not on HuggingFace, P2P delivery (step 3) enables the frontend to obtain config files automatically from the worker that has them locally. No shared filesystem or manual file copying is required — the worker serves the files on demand.

### KServe gRPC Frontend

```bash
python -m dynamo.frontend --kserve-grpc-server
```

See the [Frontend Guide](frontend-guide.md) for KServe-specific configuration and message formats.

### Kubernetes

```yaml
apiVersion: nvidia.com/v1alpha1
kind: DynamoGraphDeployment
metadata:
  name: frontend-example
spec:
  graphs:
    - name: frontend
      replicas: 1
      services:
        - name: Frontend
          image: nvcr.io/nvidia/dynamo/dynamo-vllm:latest
          command:
            - python
            - -m
            - dynamo.frontend
            - --http-port
            - "8000"
```

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--http-port` | 8000 | HTTP server port |
| `--kserve-grpc-server` | false | Enable KServe gRPC server |
| `--router-mode` | `round-robin` | Routing strategy: `round-robin`, `random`, `kv`, `direct` |

See the [Frontend Guide](frontend-guide.md) for full configuration options.

## Next Steps

| Document | Description |
|----------|-------------|
| [Configuration Reference](configuration.md) | All CLI arguments, env vars, and HTTP endpoints |
| [Frontend Guide](frontend-guide.md) | KServe gRPC configuration and integration |
| [NVIDIA Request Extensions (nvext)](nvext.md) | Custom request fields for routing hints and cache control |
| [Router Documentation](../router/README.md) | KV-aware routing configuration |
