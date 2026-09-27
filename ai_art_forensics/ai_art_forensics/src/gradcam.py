"""
Grad-CAM: highlights the image regions that most influenced the
AI-generated / authentic decision (erratic textures, blurred edges, etc.).
"""
from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


class GradCAM:
    def __init__(self, model: torch.nn.Module, target_layer_name: str | None = None):
        self.model = model
        self.gradients = None
        self.activations = None

        layer = self._find_target_layer(model, target_layer_name)
        layer.register_forward_hook(self._save_activation)
        layer.register_full_backward_hook(self._save_gradient)

    def _find_target_layer(self, model: torch.nn.Module, name: str | None):
        name = name or getattr(model, "target_layer_name", None)
        if name is None:
            raise ValueError("No target_layer_name set on model; pass one explicitly.")
        module = dict(model.named_modules()).get(name)
        if module is None:
            # fall back to the last conv-ish block for architectures with nested names
            candidates = [m for n, m in model.named_modules() if name in n]
            if not candidates:
                raise ValueError(f"Could not locate layer '{name}' in model.")
            module = candidates[-1]
        return module

    def _save_activation(self, module, inp, out):
        self.activations = out.detach()

    def _save_gradient(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def __call__(self, input_tensor: torch.Tensor, class_idx: int = 1) -> np.ndarray:
        """Returns a (H, W) heatmap in [0, 1], sized to the input tensor's H/W."""
        self.model.zero_grad(set_to_none=True)
        output = self.model(input_tensor)
        score = output[0, class_idx]
        score.backward(retain_graph=False)

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # global-avg-pool grads
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(
            cam, size=input_tensor.shape[-2:], mode="bilinear", align_corners=False
        )
        cam = cam.squeeze().cpu().numpy()
        cam -= cam.min()
        cam /= (cam.max() + 1e-8)
        return cam


def overlay_heatmap(base_img: Image.Image, cam: np.ndarray, alpha: float = 0.45) -> Image.Image:
    """Overlay a Grad-CAM heatmap (gold/obsidian colormap) onto the base image."""
    base = np.array(base_img.convert("RGB"))
    heat_u8 = np.uint8(255 * cam)
    heat_color = cv2.applyColorMap(heat_u8, cv2.COLORMAP_INFERNO)
    heat_color = cv2.cvtColor(heat_color, cv2.COLOR_BGR2RGB)

    if heat_color.shape[:2] != base.shape[:2]:
        heat_color = cv2.resize(heat_color, (base.shape[1], base.shape[0]))

    blended = (alpha * heat_color + (1 - alpha) * base).astype(np.uint8)
    return Image.fromarray(blended)


def top_activation_region(cam: np.ndarray) -> Tuple[str, str]:
    """Cheap human-readable summary of WHERE the model focused, for the forensics card."""
    h, w = cam.shape
    y, x = np.unravel_index(np.argmax(cam), cam.shape)
    v_pos = "upper" if y < h / 3 else ("lower" if y > 2 * h / 3 else "central")
    h_pos = "left" if x < w / 3 else ("right" if x > 2 * w / 3 else "middle")
    region = f"{v_pos}-{h_pos}" if v_pos != "central" or h_pos != "middle" else "central"
    intensity = float(cam.mean())
    spread_note = "tightly localized" if cam.std() > 0.22 else "diffusely spread"
    return region, spread_note
