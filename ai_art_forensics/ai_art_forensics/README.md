# Digital Atelier — AI Art Authentication Studio

A Streamlit app that classifies an uploaded painting as **Human-Made / Authentic**
or **AI-Generated / Synthetic**, with Grad-CAM visual explanations and rule-based
frequency-domain forensics, styled as a fine-art gallery / digital atelier.

## Directory structure

```
ai_art_forensics/
├── app.py                      # Streamlit UI: upload bay, dashboard, certificate export
├── train_colab.py              # Run in Colab (GPU) to fine-tune the model
├── requirements.txt
├── .streamlit/
│   └── config.toml             # gallery dark theme
├── assets/
│   └── samples/                # put 4–6 curator demo images here (see below)
├── src/
│   ├── model.py                 # backbone (ResNet-50/ConvNeXt/EfficientNet), predict()
│   ├── gradcam.py                # Grad-CAM heatmap + overlay
│   ├── forensics.py              # FFT / texture / contrast heuristics
│   ├── preprocessing.py          # robust CMYK/RGBA/corrupt-file handling
│   └── data_loader.py            # Google Drive + Kaggle dataset merging (training only)
└── weights/
    └── authenticity_model.pth    # produced by train_colab.py — NOT included here
```

## 1. Train the model (Google Colab, free GPU)

The app needs `weights/authenticity_model.pth` before it can run. Train it in Colab:

```python
!git clone <your-repo-url>
%cd <your-repo-folder>
!pip install -q -r requirements.txt
!python train_colab.py
```

`train_colab.py`:
- Mounts your Google Drive and pulls the shared folder
  `https://drive.google.com/drive/folders/1GFK26gpHu4hw6_mwpUoexBWEYN3gb3XL`
  (falls back to `gdown.download_folder` if it isn't in "My Drive").
- Downloads the supplementary Kaggle set via `kagglehub.dataset_download(
  "cashbowman/ai-generated-images-vs-real-images")`.
- Merges both into `real/` and `ai/` class folders — **inspect the printed
  folder names first** (`build_unified_dataset`'s `real_names` / `ai_names`
  sets) and adjust them to match your actual Drive folder naming.
- Fine-tunes ResNet-50 (swap `BACKBONE` in `src/model.py` for ConvNeXt-Tiny or
  EfficientNet-B3 if you prefer) and saves the best checkpoint.

Download the resulting `weights/authenticity_model.pth` from Colab (`Files` pane
→ right-click → Download) and place it in your project's `weights/` folder.

**If the weights file is over 100 MB** (GitHub's hard limit), skip committing it:
upload it instead to a public Hugging Face model repo and set an environment
variable when deploying: `MODEL_HF_REPO=yourname/art-authenticity-resnet50`.
`src/model.py` will pull it via `huggingface_hub` automatically.

## 2. Add curator demo images (so the live demo never depends on the internet)

Drop 4–6 JPGs into `assets/samples/` — a mix of well-known real paintings and
AI-diffusion pieces. Any filenames work; they're just listed and shown as
one-click "Inspect" buttons in the sidebar.

## 3. Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## 4. Zero-cost deployment

### Option A — Streamlit Community Cloud (recommended, simplest)
1. Push this repo to GitHub (public or private).
2. Go to https://share.streamlit.io → "New app" → pick the repo, branch, and
   `app.py` as the entry point.
3. In **Advanced settings → Secrets**, add (only if using the Hub fallback):
   ```
   MODEL_HF_REPO = "yourname/art-authenticity-resnet50"
   ```
4. Deploy. Free tier gives 1 GB RAM / 1 CPU — the ResNet-50 CPU forward pass
   used here (no MC-Dropout batch explosion; 8 lightweight passes) comfortably
   fits within that.

### Option B — Gradio on Hugging Face Spaces
1. Create a new Space (SDK: Gradio or Streamlit — Streamlit is supported too).
2. Upload the same repo contents; if using Streamlit SDK, no code changes
   needed. Set `MODEL_HF_REPO` as a Space secret if weights live in a
   separate model repo (recommended: keep both in the same HF account, no
   git-lfs headaches with the 100 MB GitHub limit).

### Option C — FastAPI + React (decoupled, more setup)
Only worth it if you want a fully custom frontend. Wrap `src/model.py`,
`src/gradcam.py`, and `src/forensics.py` behind three FastAPI routes
(`/predict`, `/gradcam`, `/forensics`), deploy the API on Render's free web
service tier, and host a Vite/React frontend consuming it on Vercel or
Netlify. This roughly triples the implementation surface for a demo whose
audience only sees Option A's output anyway — recommended only if the
brief explicitly requires a decoupled architecture.

## Notes on the classifier itself

- This scaffold ships **architecture + pipeline**, not pretrained weights —
  you must run `train_colab.py` against real data before predictions mean
  anything. Until then, `app.py` will show a clear "weights not found" error
  instead of failing silently.
- MC-Dropout (8 stochastic passes) is used only to produce the confidence
  *band* shown in the UI — it is a lightweight uncertainty proxy, not a
  substitute for a properly calibrated model (e.g. temperature scaling) if
  you extend this beyond a class demo.
- The FFT / Laplacian / local-contrast checks in `src/forensics.py` are
  heuristics for the "why" narrative on the report card — they are not the
  classifier and can disagree with the model's verdict on edge cases; that's
  expected and worth mentioning live if an evaluator asks about it.
