"""
image_io.py — VERBATIM copy of the real subnet's perturbnet/image_io.py.

This is the exact image <-> tensor pipeline the validator and miner use. Two facts that
drive the whole attack design live here:

  * decode_image_b64 returns a CHW float tensor in [0,1] at the image's NATIVE resolution
    (no resize/crop). This is the space L-infinity / RMSE / SSIM / PSNR are measured in.
  * encode_image_b64 rounds to 8-bit PNG (.round().astype(uint8)). So whatever the miner
    sends is quantised to multiples of 1/255 (~0.00392) before the validator ever sees it.
"""

from __future__ import annotations

import base64
import io

import numpy as np
import torch
from PIL import Image


def decode_image_b64(image_b64: str) -> torch.Tensor:
    raw = base64.b64decode(image_b64)
    image = Image.open(io.BytesIO(raw)).convert("RGB")
    arr = np.asarray(image, dtype=np.float32) / 255.0          # uint8 -> [0,1], native resolution
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()  # HWC -> CHW


def encode_image_b64(image_chw: torch.Tensor) -> str:
    clipped = image_chw.detach().cpu().clamp(0.0, 1.0)
    arr = (clipped.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)  # 8-bit quantise
    image = Image.fromarray(arr, mode="RGB")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")
