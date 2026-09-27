# Classification-between-AI-slop-and-real-art
An AI-powered forensic tool that distinguishes authentic, human-made art from synthetic, AI-generated imagery.
# 🎨 Canvas Forensics: Art Authentication & AI Detection

Welcome to **Canvas Forensics**, an academic AI project designed to solve a modern digital problem: distinguishing authentic, human-created paintings from synthetic, AI-generated art (often referred to as "AI slop"). 

Packaged in an elegant, gallery-themed web application, this project doesn't just output a "Real" or "Fake" label. It utilizes **Explainable AI (XAI)** to show exactly *why* a decision was made, acting as a digital art curator and forensic analyst.

### ✨ Key Features
* **Deep Learning Classification:** Built on a transfer-learning vision backbone (e.g., ResNet-50) fine-tuned on custom datasets of traditional masterpieces and modern diffusion-generated images.
* **Explainable AI (Grad-CAM):** Generates transparent visual heatmaps overlaid on the artwork, highlighting the exact brushstrokes, erratic high-frequency noise, or anatomical artifacts that triggered the model's prediction.
* **Forensic Heuristics:** Analyzes frequency domain artifacts (via FFT) and texture variance to break down the synthetic signatures common in AI generation.
* **Museum Certificate of Analysis:** A beautiful, exportable forensic report detailing the artwork's authenticity probability and detected anomalies.
* **Curator's Showcase:** A built-in, offline-safe demo bay for seamless live presentations and evaluations.

### 🛠️ Tech Stack
* **Machine Learning:** PyTorch, torchvision, OpenCV
* **UI/UX:** Streamlit (styled with a custom Fine Art Gallery aesthetic)
* **Data Processing:** Kaggle Datasets & Google Drive Integration
