"""
Run this in Google Colab (GPU runtime) to fine-tune the authenticity classifier.
Not part of the deployed Streamlit app -- produces weights/authenticity_model.pth,
which you then commit (or upload to Hugging Face Hub if >100MB) for the app to load.

Colab quick start:
    !pip install -q kagglehub gdown
    !git clone <your-repo-url> && cd <your-repo>
    !python train_colab.py
"""
from __future__ import annotations

import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from src.data_loader import build_unified_dataset, fetch_kaggle_supplement, mount_drive_and_fetch
from src.model import build_model

IMAGE_SIZE = 384
BATCH_SIZE = 32
EPOCHS = 12
LR = 3e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main() -> None:
    print("Device:", DEVICE)

    # 1) Pull data from both sources and merge into ImageFolder layout.
    drive_root = mount_drive_and_fetch()
    kaggle_root = fetch_kaggle_supplement()
    data_root = build_unified_dataset(drive_root=drive_root, kaggle_root=kaggle_root)

    train_tf = transforms.Compose(
        [
            transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.85, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.1, 0.1, 0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    val_tf = transforms.Compose(
        [
            transforms.Resize(int(IMAGE_SIZE * 1.14)),
            transforms.CenterCrop(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    full_ds = datasets.ImageFolder(data_root, transform=train_tf)
    print("Classes (index order should now be explicitly 0_real and 1_ai):")
    print(full_ds.class_to_idx)

    n_val = max(1, int(0.15 * len(full_ds)))
    n_train = len(full_ds) - n_val
    train_ds, val_ds = random_split(full_ds, [n_train, n_val])
    val_ds.dataset.transform = val_tf  # use non-augmented transform for validation

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    model = build_model(pretrained=True).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_val_acc = 0.0
    os.makedirs("weights", exist_ok=True)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * imgs.size(0)
        scheduler.step()

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                preds = model(imgs).argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        val_acc = correct / max(1, total)
        print(f"Epoch {epoch}/{EPOCHS} | train_loss={running_loss/len(train_ds):.4f} | val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "weights/authenticity_model.pth")
            print(f"  -> saved new best ({val_acc:.4f})")

    print("Done. Best val accuracy:", best_val_acc)
    print("Weights at weights/authenticity_model.pth -- commit this file, or upload it to")
    print("the Hugging Face Hub and set MODEL_HF_REPO if it exceeds GitHub's 100MB limit.")


if __name__ == "__main__":
    main()
