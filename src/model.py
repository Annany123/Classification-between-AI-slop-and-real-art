"""
Backbone: ResNet-50 (ImageNet pretrained) fine-tuned as a binary
authenticity classifier: 0 = Human-Made / Authentic, 1 = AI-Generated / Synthetic.

Swap BACKBONE to "convnext_tiny" or "efficientnet_b3" if you fine-tune one of
those instead -- build_model() handles all three.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
from torchvision import models

BACKBONE = "resnet50"  # one of: resnet50, convnext_tiny, efficientnet_b3
NUM_CLASSES = 2
CLASS_NAMES = ["Human-Made / Authentic", "AI-Generated / Synthetic"]

DEFAULT_WEIGHTS_PATH = os.environ.get("MODEL_WEIGHTS_PATH", "weights/authenticity_model.pth")
# If the .pth exceeds GitHub's 100MB limit, host it on the Hub and set this instead:
HF_HUB_REPO = os.environ.get("MODEL_HF_REPO", "")  # e.g. "yourname/art-authenticity-resnet50"


def build_model(pretrained: bool = True) -> nn.Module:
    if BACKBONE == "resnet50":
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        net = models.resnet50(weights=weights)
        net.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(net.fc.in_features, NUM_CLASSES),
        )
        target_layer_name = "layer4"
    elif BACKBONE == "convnext_tiny":
        weights = models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
        net = models.convnext_tiny(weights=weights)
        net.classifier[2] = nn.Linear(net.classifier[2].in_features, NUM_CLASSES)
        target_layer_name = "features"
    elif BACKBONE == "efficientnet_b3":
        weights = models.EfficientNet_B3_Weights.IMAGENET1K_V1 if pretrained else None
        net = models.efficientnet_b3(weights=weights)
        net.classifier[1] = nn.Linear(net.classifier[1].in_features, NUM_CLASSES)
        target_layer_name = "features"
    else:
        raise ValueError(f"Unknown BACKBONE: {BACKBONE}")

    net.target_layer_name = target_layer_name  # stashed for Grad-CAM lookup
    return net


@dataclass
class Prediction:
    label: str
    ai_probability: float          # 0-1
    authentic_probability: float   # 0-1
    confidence_low: float          # lower bound of a simple uncertainty band
    confidence_high: float


_MODEL_CACHE: dict = {}


def _resolve_weights_path() -> str:
    if os.path.exists(DEFAULT_WEIGHTS_PATH):
        return DEFAULT_WEIGHTS_PATH
    if HF_HUB_REPO:
        from huggingface_hub import hf_hub_download

        return hf_hub_download(repo_id=HF_HUB_REPO, filename="authenticity_model.pth")
    raise FileNotFoundError(
        f"No model weights found at '{DEFAULT_WEIGHTS_PATH}' and MODEL_HF_REPO is not set. "
        "Run train_colab.py first, or point MODEL_WEIGHTS_PATH / MODEL_HF_REPO at your weights."
    )


def load_model(device: str = "cpu") -> nn.Module:
    """Cached model loader so Streamlit doesn't reload weights on every rerun."""
    if "model" in _MODEL_CACHE:
        return _MODEL_CACHE["model"]

    net = build_model(pretrained=False)
    weights_path = _resolve_weights_path()
    state_dict = torch.load(weights_path, map_location=device)
    net.load_state_dict(state_dict)
    net.to(device)
    net.eval()
    _MODEL_CACHE["model"] = net
    return net


@torch.no_grad()
def predict(net: nn.Module, tensor: torch.Tensor, device: str = "cpu",
            mc_dropout_passes: int = 8) -> Prediction:
    """Forward pass with lightweight MC-Dropout to produce an uncertainty band.

    Keeps the Dropout layer active across a few stochastic passes so the UI can
    show a confidence range rather than a single point estimate.
    """
    tensor = tensor.to(device)

    net.eval()
    for m in net.modules():
        if isinstance(m, nn.Dropout):
            m.train()  # re-enable dropout only, for MC sampling

    probs = []
    for _ in range(mc_dropout_passes):
        logits = net(tensor)
        probs.append(torch.softmax(logits, dim=1)[0, 1].item())  # P(AI-generated)

    net.eval()  # restore full eval mode

    ai_prob = sum(probs) / len(probs)
    spread = (max(probs) - min(probs)) / 2
    label = CLASS_NAMES[1] if ai_prob >= 0.5 else CLASS_NAMES[0]

    return Prediction(
        label=label,
        ai_probability=ai_prob,
        authentic_probability=1 - ai_prob,
        confidence_low=max(0.0, ai_prob - spread),
        confidence_high=min(1.0, ai_prob + spread),
    )
