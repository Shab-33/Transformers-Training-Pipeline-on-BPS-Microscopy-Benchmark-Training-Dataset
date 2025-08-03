# Vision Transformer on NASA BPS Microscopy Data

This repo contains my project where I fine-tune a Vision Transformer (ViT) to classify cell damage from NASA’s BPS microscopy dataset. The goal is to teach the model to distinguish between **Fe** and **X-ray** radiation damage using attention-based explainability techniques like SHAP and rollout heatmaps.

---

## What’s Inside

This repo includes:
- A full training pipeline using Hugging Face's `transformers`
- Validation and test evaluation with accuracy reporting
- Visual explanations using:
  - Attention rollout (to see where the model looks)
  - SHAP heatmaps (to understand feature importance)

Everything’s in the main Colab notebook. You can run the full process end-to-end or jump to just the testing and visualisation parts.

---

## Dataset

I’ve already included the **sorted BPS dataset** in this repo. Here’s what I did:

- Grouped all images into two classes: **Fe** and **X-ray**
- Split the dataset into:
  - **70%** training
  - **15%** validation
  - **15%** testing
- Each split is inside its own folder and ready to use with `torchvision.datasets.ImageFolder`

No extra data prep needed — just point the notebook to the folder and go.

---

## Quick Start (in Colab)

1. Open the notebook in Google Colab  
2. Mount your Google Drive (used to store/save model checkpoints)  
3. Run through each section:
   - Training
   - Evaluation
   - Attention + SHAP visualisations

---

## Goal

Besides getting good accuracy, the aim is to **understand why the model predicts what it does**, using visual tools to interpret its focus on cell structures.

Example Images From VIT Base:

<img width="1907" height="698" alt="download (1)" src="https://github.com/user-attachments/assets/479334c4-39e5-4c3d-b99d-0aee2276e4c4" />
<img width="1907" height="698" alt="download (2)" src="https://github.com/user-attachments/assets/5fa288f3-7a3f-4704-bd09-1c44ab785555" />
<img width="717" height="364" alt="download (3)" src="https://github.com/user-attachments/assets/f98495d7-6434-44f5-8964-fe20747fb2ba" />
