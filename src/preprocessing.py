"""
Robust image preprocessing for the Artwork Inspection Bay.
Handles arbitrary user uploads: CMYK/RGBA/P mode, corrupt files, odd aspect ratios.
"""
from __future__ import annotations

import io
from typing import Tuple

import numpy as np
import torch
from PIL import Image, ImageOps
from torchvision import transforms

IMAGE_SIZE = 384  # ConvNeXt-Tiny native-ish input size; drop to 224 for ResNet-50

_MEAN = [0.485, 0.456, 0.406]
_STD = [0.229, 0.224, 0.225]

_inference_transform = transforms.Compose(
    [
        transforms.Resize(int(IMAGE_SIZE * 1.14)),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=_MEAN, std=_STD),
    ]
)


class UnreadableImageError(ValueError):
    """Raised when an uploaded file cannot be decoded as an image."""


def load_and_normalize(file_bytes: bytes) -> Image.Image:
    """Decode arbitrary uploaded bytes into a clean RGB PIL Image.

    Handles: CMYK, RGBA (flattened onto white), palette (P) mode, EXIF
    orientation, and truncated/corrupt files (raises UnreadableImageError).
    """
    try:
        img = Image.open(io.BytesIO(file_bytes))
        img.load()
    except Exception as exc:  # noqa: BLE001 - want to catch any PIL decode failure
        raise UnreadableImageError(f"Could not decode image: {exc}") from exc

    img = ImageOps.exif_transpose(img)  # respect camera/scanner orientation

    if img.mode == "CMYK":
        img = img.convert("RGB")
    elif img.mode in ("RGBA", "LA", "P"):
        background = Image.new("RGB", img.size, (255, 255, 255))
        rgba = img.convert("RGBA")
        background.paste(rgba, mask=rgba.split()[-1])
        img = background
    elif img.mode != "RGB":
        img = img.convert("RGB")

    return img


def to_model_tensor(img: Image.Image) -> torch.Tensor:
    """PIL RGB image -> normalized (1, 3, IMAGE_SIZE, IMAGE_SIZE) tensor."""
    tensor = _inference_transform(img)
    return tensor.unsqueeze(0)


def to_display_square(img: Image.Image, size: int = IMAGE_SIZE) -> Image.Image:
    """Resize + center-crop to a square for consistent heatmap overlay alignment."""
    w, h = img.size
    scale = (size * 1.14) / min(w, h)
    resized = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS)
    rw, rh = resized.size
    left = (rw - size) // 2
    top = (rh - size) // 2
    return resized.crop((left, top, left + size, top + size))


def prepare(file_bytes: bytes) -> Tuple[Image.Image, torch.Tensor]:
    """One-call pipeline: raw bytes -> (display image, model tensor)."""
    clean = load_and_normalize(file_bytes)
    display_img = to_display_square(clean)
    tensor = to_model_tensor(clean)
    return display_img, tensor
