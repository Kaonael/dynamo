# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import Any, Dict

import torch

logger = logging.getLogger(__name__)


def get_kimi_k25_image_features(
    vision_encoder: torch.nn.Module, image_embeds: Dict[str, Any]
) -> torch.Tensor:
    """
    Extract image features using MoonViT-3D vision encoder for Kimi K2.5.

    Kimi K2.5 uses the MoonViT-3D (400M params) vision encoder which processes
    images at up to 4K resolution. The encoder produces patch tokens that are
    projected into the language model's embedding space.

    Args:
        vision_encoder: The MoonViT-3D vision encoder model
        image_embeds: Dictionary containing pixel values from the image processor

    Returns:
        Processed image features tensor
    """
    pixel_values = image_embeds["pixel_values"].to(vision_encoder.device)

    # MoonViT-3D uses image_grid_thw for temporal/height/width grid info
    grid_thw = image_embeds.get("image_grid_thw", None)
    if grid_thw is not None:
        grid_thw = grid_thw.to(vision_encoder.device)
        logger.debug(f"Kimi K2.5 grid_thw shape: {grid_thw.shape}")

    if hasattr(vision_encoder, "get_image_features"):
        if grid_thw is not None:
            return vision_encoder.get_image_features(pixel_values, grid_thw)
        else:
            return vision_encoder.get_image_features(pixel_values)
    else:
        # Fallback: direct forward pass through the vision model
        outputs = vision_encoder(pixel_values)
        if hasattr(outputs, "last_hidden_state"):
            return outputs.last_hidden_state
        return outputs
