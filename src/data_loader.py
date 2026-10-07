"""
Dataset loading for TRAINING ONLY (run inside Google Colab, not in the deployed
Streamlit app). Two sources are supported and can be combined:

  1. Your Google Drive folder of curated paintings / AI art:
     https://drive.google.com/drive/folders/1GFK26gpHu4hw6_mwpUoexBWEYN3gb3XL

  2. The public Kaggle set for extra volume:
     kagglehub: "cashbowman/ai-generated-images-vs-real-images"

Both are normalized into a single flat structure:
    <root>/0_real/*.jpg
    <root>/1_ai/*.jpg
so torchvision.datasets.ImageFolder can consume it directly and explicitly map 0 to real and 1 to ai.
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path

DRIVE_FOLDER_ID = "1GFK26gpHu4hw6_mwpUoexBWEYN3gb3XL"
DRIVE_URL = f"https://drive.google.com/drive/folders/{DRIVE_FOLDER_ID}"


def mount_drive_and_fetch(dest_dir: str = "/content/data/drive_art") -> str:
    """Run inside Colab. Mounts Drive, then copies the shared folder locally.

    Expects the Drive folder to already contain two subfolders named
    'real' and 'ai' (rename on Drive if yours differ, or adjust the map below).
    """
    from google.colab import drive  # noqa: E402  (Colab-only import)

    drive.mount("/content/drive")

    # If the folder is a *shared* folder (not in "My Drive"), locate it under
    # "Shared drives" or "Shared with me" mount paths; adjust this glob if needed.
    candidates = list(Path("/content/drive").rglob("*"))
    matches = [p for p in candidates if p.is_dir() and DRIVE_FOLDER_ID in str(p)]
    src = matches[0] if matches else None

    if src is None:
        # Fallback: use gdown to pull the folder by ID directly (works for
        # "Anyone with the link" shared folders without needing drive.mount).
        import gdown

        os.makedirs(dest_dir, exist_ok=True)
        gdown.download_folder(url=DRIVE_URL, output=dest_dir, quiet=False, use_cookies=False)
        return dest_dir

    os.makedirs(dest_dir, exist_ok=True)
    shutil.copytree(src, dest_dir, dirs_exist_ok=True)
    return dest_dir


def fetch_kaggle_supplement() -> str:
    """Downloads the supplementary Kaggle dataset via kagglehub (as provided)."""
    import kagglehub

    path = kagglehub.dataset_download("cashbowman/ai-generated-images-vs-real-images")
    print("Kaggle dataset cached at:", path)
    return path


def _flatten_into(class_dir: Path, sources: list[Path], exts=(".jpg", ".jpeg", ".png")) -> None:
    class_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for src_dir in sources:
        if not src_dir.exists():
            continue
        for f in src_dir.rglob("*"):
            if f.suffix.lower() in exts:
                target = class_dir / f"{src_dir.name}_{n}{f.suffix.lower()}"
                if not target.exists():
                    shutil.copy2(f, target)
                n += 1
    print(f"  -> {class_dir}: {n} files")


def build_unified_dataset(
    output_root: str = "/content/data/unified",
    drive_root: str | None = None,
    kaggle_root: str | None = None,
) -> str:
    """Merge Drive + Kaggle sources into ImageFolder-ready real/ and ai/ dirs.

    Adjust the guessed subfolder names below (`real_names` / `ai_names`) to
    match however your specific Drive folder and the Kaggle set are laid out --
    inspect the downloaded paths first with `!find <root> -maxdepth 2`.
    """
    output_root = Path(output_root)
    real_names = {"real", "authentic", "human", "paintings", "real_images"}
    ai_names = {"ai", "fake", "synthetic", "ai_generated", "generated"}

    sources = [p for p in (drive_root, kaggle_root) if p]
    real_dirs, ai_dirs = [], []
    for root in sources:
        for sub in Path(root).rglob("*"):
            if sub.is_dir() and sub.name.lower() in real_names:
                real_dirs.append(sub)
            elif sub.is_dir() and sub.name.lower() in ai_names:
                ai_dirs.append(sub)

    print("Real-class source dirs found:", real_dirs)
    print("AI-class source dirs found:", ai_dirs)

    _flatten_into(output_root / "0_real", real_dirs)
    _flatten_into(output_root / "1_ai", ai_dirs)

    return str(output_root)
