"""
Digital Atelier — AI Art Authentication Studio
Run: streamlit run app.py
"""
from __future__ import annotations

import io
import os
import time
from pathlib import Path

import streamlit as st
from PIL import Image, ImageDraw, ImageFont

from src.forensics import analyze
from src.gradcam import GradCAM, overlay_heatmap, top_activation_region
from src.model import CLASS_NAMES, load_model, predict
from src.preprocessing import UnreadableImageError, prepare

# --------------------------------------------------------------------------- #
# Page setup + gallery CSS
# --------------------------------------------------------------------------- #
st.set_page_config(page_title="Digital Atelier — Art Authentication", page_icon="🖼️", layout="wide")

GOLD = "#D4AF37"
OBSIDIAN = "#0F1115"
SLATE = "#1B1E24"
PARCHMENT = "#F5F2EB"

st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@500;700&family=Inter:wght@400;500;600&display=swap');

    .stApp {{ background-color: {OBSIDIAN}; color: {PARCHMENT}; }}
    h1, h2, h3 {{ font-family: 'Cormorant Garamond', serif !important; color: {PARCHMENT}; letter-spacing: 0.02em; }}
    .gallery-title {{ font-size: 3rem; font-weight: 700; border-bottom: 1px solid {GOLD}44; padding-bottom: .4rem; }}
    .gallery-sub {{ font-family: 'Inter', sans-serif; color: {GOLD}; letter-spacing: .18em; text-transform: uppercase; font-size: .78rem; }}
    div[data-testid="stFileUploaderDropzone"] {{
        background-color: {SLATE}; border: 1.5px dashed {GOLD}88 !important; border-radius: 6px;
    }}
    .forensics-card {{
        background: linear-gradient(180deg, {SLATE} 0%, #14161c 100%);
        border: 1px solid {GOLD}55; border-radius: 10px; padding: 1.4rem 1.6rem;
        font-family: 'Inter', sans-serif;
    }}
    .verdict-ai {{ color: #E4664B; font-weight: 600; }}
    .verdict-real {{ color: #7FBF8F; font-weight: 600; }}
    .scan-caption {{ font-family: 'Inter', sans-serif; color: {GOLD}; letter-spacing: .12em; font-size: .75rem; text-transform: uppercase; }}
    .stButton>button {{
        background-color: transparent; border: 1px solid {GOLD}; color: {GOLD};
        font-family: 'Inter', sans-serif; letter-spacing: .08em; text-transform: uppercase; font-size: .78rem;
    }}
    .stButton>button:hover {{ background-color: {GOLD}22; color: {PARCHMENT}; border-color: {GOLD}; }}
    </style>
    """,
    unsafe_allow_html=True,
)

DEVICE = "cpu"
SAMPLES_DIR = Path(__file__).parent / "assets" / "samples"

# --------------------------------------------------------------------------- #
# Header
# --------------------------------------------------------------------------- #
st.markdown('<div class="gallery-sub">Digital Atelier · Forensic Wing</div>', unsafe_allow_html=True)
st.markdown('<div class="gallery-title">Artwork Authentication Studio</div>', unsafe_allow_html=True)
st.write(
    "Upload a painting for forensic inspection. The atelier will render a Grad-CAM "
    "diagnostic, run frequency-domain heuristics, and issue a Certificate of Analysis."
)

# --------------------------------------------------------------------------- #
# Model (cached across reruns)
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner=False)
def get_model():
    return load_model(device=DEVICE)


model_load_error = None
try:
    model = get_model()
except Exception as exc:  # noqa: BLE001
    model_load_error = str(exc)
    model = None

# --------------------------------------------------------------------------- #
# Artwork Inspection Bay
# --------------------------------------------------------------------------- #
left, right = st.columns([1.1, 1])

with left:
    st.markdown("#### The Inspection Bay")
    uploaded = st.file_uploader("Drag a painting here (JPG / PNG)", type=["jpg", "jpeg", "png", "webp"])

    st.markdown("###### or select from the Curator's Sample Showcase")
    sample_files = sorted(SAMPLES_DIR.glob("*")) if SAMPLES_DIR.exists() else []
    chosen_sample = None
    if sample_files:
        cols = st.columns(min(len(sample_files), 6))
        for i, sf in enumerate(sample_files[:6]):
            with cols[i % len(cols)]:
                st.image(str(sf), use_container_width=True)
                if st.button("Inspect", key=f"sample_{i}"):
                    chosen_sample = sf
    else:
        st.caption(
            "No curator samples bundled yet — drop 4–6 JPGs into "
            "`assets/samples/` (mix of real paintings and AI-generated pieces) "
            "so the live demo never depends on an internet connection."
        )

file_bytes = None
source_label = None
if uploaded is not None:
    file_bytes = uploaded.getvalue()
    source_label = uploaded.name
elif chosen_sample is not None:
    file_bytes = chosen_sample.read_bytes()
    source_label = chosen_sample.name

# --------------------------------------------------------------------------- #
# Analysis Dashboard
# --------------------------------------------------------------------------- #
with right:
    st.markdown("#### Analysis Dashboard")

    if model_load_error:
        st.error(
            "Model weights are not loaded yet. Run `train_colab.py` to produce "
            "`weights/authenticity_model.pth`, or set MODEL_HF_REPO to a Hugging "
            f"Face repo hosting the weights.\n\nDetail: {model_load_error}"
        )
    elif file_bytes is None:
        st.info("Awaiting an artwork — upload a file or choose a curator sample.")
    else:
        try:
            display_img, tensor = prepare(file_bytes)
        except UnreadableImageError as exc:
            st.error(f"Could not process this file: {exc}")
            st.stop()

        scan_placeholder = st.empty()
        with scan_placeholder:
            st.markdown('<div class="scan-caption">Forensic scanning…</div>', unsafe_allow_html=True)
            st.progress(0)
            for pct in (25, 55, 80, 100):
                time.sleep(0.12)
                st.progress(pct)
        scan_placeholder.empty()

        tensor.requires_grad_(False)
        cam_tensor = tensor.clone().requires_grad_(True)
        gradcam = GradCAM(model)
        cam = gradcam(cam_tensor, class_idx=1)  # class 1 = AI-generated
        heatmap_img = overlay_heatmap(display_img, cam)

        pred = predict(model, tensor, device=DEVICE)
        report = analyze(display_img)
        region, spread_note = top_activation_region(cam)

        verdict_class = "verdict-ai" if pred.ai_probability >= 0.5 else "verdict-real"
        st.markdown(
            f'<div class="forensics-card">'
            f'<div class="scan-caption">Predicted label</div>'
            f'<h3 class="{verdict_class}">{pred.label}</h3>'
            f'<b>AI-Generated Probability:</b> {pred.ai_probability*100:.1f}%'
            f' &nbsp;(range {pred.confidence_low*100:.1f}–{pred.confidence_high*100:.1f}%)<br>'
            f'<b>Authentic Probability:</b> {pred.authentic_probability*100:.1f}%'
            f'</div>',
            unsafe_allow_html=True,
        )

        st.markdown("###### Dual-View Inspection")
        slider_pct = st.slider("Original ↔ Grad-CAM diagnostic", 0, 100, 50, label_visibility="collapsed")
        blended = Image.blend(display_img.convert("RGB"), heatmap_img.convert("RGB"), slider_pct / 100)
        st.image(blended, use_container_width=True, caption=f"{source_label}")

        st.markdown("###### Provenance & Forensics Card")
        st.markdown(
            f"""
            <div class="forensics-card">
            <b>Grad-CAM focus:</b> {region} region, {spread_note}<br>
            <b>Frequency-domain:</b> {report.fft_verdict} (high-freq ratio {report.fft_high_freq_ratio:.3f})<br>
            <b>Micro-texture:</b> {report.smoothness_verdict} (Laplacian var {report.laplacian_variance:.5f})<br>
            <b>Local contrast:</b> {report.contrast_verdict} (std {report.local_contrast_std:.4f})<br><br>
            <b>Notes:</b> {' '.join(report.notes)}
            </div>
            """,
            unsafe_allow_html=True,
        )

        # ------------------------------------------------------------- #
        # Museum Certificate of Analysis (downloadable)
        # ------------------------------------------------------------- #
        def build_certificate() -> bytes:
            cert = Image.new("RGB", (1000, 700), PARCHMENT)
            draw = ImageDraw.Draw(cert)
            try:
                title_font = ImageFont.truetype(
                    "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf", 34
                )
                body_font = ImageFont.truetype(
                    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18
                )
            except OSError:
                title_font = ImageFont.load_default()
                body_font = ImageFont.load_default()

            draw.rectangle([20, 20, 980, 680], outline=(212, 175, 55), width=3)
            draw.text((60, 60), "CERTIFICATE OF ANALYSIS", font=title_font, fill=(20, 20, 24))
            draw.text((60, 110), "Digital Atelier — Forensic Wing", font=body_font, fill=(80, 80, 80))
            thumb = display_img.copy()
            thumb.thumbnail((320, 320))
            cert.paste(thumb, (60, 170))

            lines = [
                f"Artwork file: {source_label}",
                f"Predicted label: {pred.label}",
                f"AI-generated probability: {pred.ai_probability*100:.1f}%",
                f"Confidence band: {pred.confidence_low*100:.1f}% – {pred.confidence_high*100:.1f}%",
                "",
                f"Grad-CAM focus: {region}, {spread_note}",
                f"Frequency-domain: {report.fft_verdict}",
                f"Micro-texture: {report.smoothness_verdict}",
                f"Local contrast: {report.contrast_verdict}",
            ]
            y = 175
            for line in lines:
                draw.text((410, y), line, font=body_font, fill=(20, 20, 24))
                y += 34

            buf = io.BytesIO()
            cert.save(buf, format="PNG")
            return buf.getvalue()

        st.download_button(
            "Export Certificate of Analysis",
            data=build_certificate(),
            file_name=f"certificate_{Path(source_label).stem}.png",
            mime="image/png",
        )
