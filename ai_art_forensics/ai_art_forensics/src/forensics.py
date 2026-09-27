"""
Heuristic, model-independent signals that complement Grad-CAM: diffusion models
tend to leave characteristic fingerprints in the frequency domain and in local
texture statistics. These are descriptive signals for the report card, not the
classifier itself.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image
from skimage.color import rgb2gray
from skimage.filters import laplace


@dataclass
class ForensicReport:
    fft_high_freq_ratio: float       # share of spectral energy in the high-freq band
    fft_verdict: str
    laplacian_variance: float        # edge/texture sharpness proxy
    smoothness_verdict: str
    local_contrast_std: float
    contrast_verdict: str
    notes: list[str]


def _fft_high_frequency_ratio(gray: np.ndarray) -> float:
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude = np.abs(fshift)

    h, w = gray.shape
    cy, cx = h // 2, w // 2
    radius = min(h, w) // 6  # inner "low frequency" disk

    yy, xx = np.ogrid[:h, :w]
    dist = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    low_mask = dist <= radius
    high_mask = ~low_mask

    total_energy = magnitude.sum() + 1e-8
    high_energy = magnitude[high_mask].sum()
    return float(high_energy / total_energy)


def analyze(img: Image.Image) -> ForensicReport:
    arr = np.array(img.convert("RGB")).astype(np.float32) / 255.0
    gray = rgb2gray(arr)

    hf_ratio = _fft_high_frequency_ratio(gray)
    # Diffusion/GAN upsampling artifacts often show either unnaturally suppressed
    # high-frequency detail (over-smoothed) or a sharp, regular high-freq spike
    # (checkerboard/upsampling artifacts). Traditional media + camera noise sits
    # in a broader mid-band.
    if hf_ratio < 0.12:
        fft_verdict = "Unusually smooth spectrum — consistent with diffusion up-sampling"
    elif hf_ratio > 0.35:
        fft_verdict = "Sharp regular high-frequency spike — possible GAN/upsampling artifact"
    else:
        fft_verdict = "Broad natural frequency spread — consistent with organic media/sensor noise"

    lap_var = float(laplace(gray).var())
    if lap_var < 0.0008:
        smoothness_verdict = "Low micro-texture variance — brushwork/canvas grain largely absent"
    else:
        smoothness_verdict = "Textural micro-variance present — consistent with physical media"

    # Local contrast: split into patches, look at std of patch-level std (blockiness)
    patch = 16
    h, w = gray.shape
    h2, w2 = (h // patch) * patch, (w // patch) * patch
    trimmed = gray[:h2, :w2]
    blocks = trimmed.reshape(h2 // patch, patch, w2 // patch, patch).swapaxes(1, 2)
    block_stds = blocks.reshape(-1, patch, patch).std(axis=(1, 2))
    contrast_std = float(block_stds.std())

    if contrast_std < 0.02:
        contrast_verdict = "Very uniform local contrast — atypical for hand-applied pigment"
    else:
        contrast_verdict = "Irregular local contrast — consistent with manual paint application"

    notes = []
    if "diffusion" in fft_verdict or "GAN" in fft_verdict:
        notes.append("Frequency-domain signature leans synthetic.")
    if "absent" in smoothness_verdict:
        notes.append("Lacks expected physical-media micro-texture.")
    if "atypical" in contrast_verdict:
        notes.append("Local contrast distribution is suspiciously uniform.")
    if not notes:
        notes.append("No strong synthetic-artifact signals detected by heuristics.")

    return ForensicReport(
        fft_high_freq_ratio=hf_ratio,
        fft_verdict=fft_verdict,
        laplacian_variance=lap_var,
        smoothness_verdict=smoothness_verdict,
        local_contrast_std=contrast_std,
        contrast_verdict=contrast_verdict,
        notes=notes,
    )
