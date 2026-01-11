# Multi-level Cognitive Architecture for Multimodal Tumor Analysis

This repository implements a **three-level cognitive architecture** for tumor
analysis and classification, integrating automated ROI segmentation, multimodal
feature extraction, and transformer-based feature fusion.

The framework is designed for **medical imaging applications** and emphasizes
modularity, interpretability, and reproducibility.

---

## 🧠 Overall Architecture

The proposed framework consists of three hierarchical modules:

Level I : ROI Automatic Segmentation (2D U-Net)
Level II : Multimodal Feature Extraction
Level III : ViT-based Multidimensional Feature Fusion and Classification

Each module is implemented (or documented) as an independent and reusable
component.

---

## 📂 Repository Structure

├─ A_ROI_Segmentation_2DUNet/
├─ B_Multimodal_Feature_Extraction/
├─ C_ViT_Fusion/
└─ README.md

---

## 🔹 Module A: ROI Automatic Segmentation (2D U-Net)

**Path**: `A_ROI_Segmentation_2DUNet/`

Module A performs **automatic tumor ROI segmentation** from 3D medical images
(NIfTI format) using a **2D U-Net** trained on slice-wise data.

### Key Features
- 3D NIfTI → 2D slice-based training
- Combined **BCEWithLogits + Dice loss**
- Data augmentation (flip, rotation)
- Inference with per-slice **overlay / probability / binary mask PNG outputs**

### Directory Overview
A_ROI_Segmentation_2DUNet/
├─ src/
│ ├─ datasets/ # NIfTI slice dataset
│ ├─ models/ # 2D U-Net
│ ├─ losses/ # Dice loss
│ ├─ utils/ # Visualization tools
│ ├─ train_seg.py
│ └─ infer_seg_png.py
├─ docs/
│ ├─ data_format.md # Data organization & mask definition
│ └─ demo_results/ # Example PNG results

> ⚠️ Raw medical images (`.nii.gz`) and trained model weights are **not included**
> in this repository. See `docs/data_format.md` for data preparation details.

---

## 🔹 Module B: Multimodal Feature Extraction

**Path**: `B_Multimodal_Feature_Extraction/`

Module B corresponds to the **second-level cognitive architecture** and extracts
heterogeneous features from segmented tumor regions.

### Extracted Feature Modalities
1. **Radiomics features** (PyRadiomics)
2. **2D CNN features** (DenseNet121)
3. **3D CNN features** (ShuffleNet3D)
4. **Clinical variables** (structured tabular data)

### Implementation Note
- Radiomics extraction is based on the open-source **PyRadiomics** library.
- Deep learning feature extraction (DenseNet121, ShuffleNet3D) is implemented
  using **vendor-provided (onekey) code**, which is proprietary.
- Due to licensing restrictions, **source code for Module B is not redistributed**.

### What *Is* Provided
- Detailed methodological descriptions
- Feature definitions and preprocessing notes
- Licensing and reproducibility statements

B_Multimodal_Feature_Extraction/
├─ README.md
└─ docs/
├─ overview.md
├─ radiomics_features.md
├─ deep_features.md
├─ clinical_variables.md
└─ licensing_and_disclaimer.md

> ✅ Any alternative implementation that produces **equivalent feature tables**
> can be used to reproduce downstream results.

---

## 🔹 Module C: ViT-based Multimodal Feature Fusion

**Path**: `C_ViT_Fusion/`

Module C implements the **core methodological contribution** of this work:
a **Vision Transformer (ViT)–based fusion network** that models cross-modality
relationships and performs final classification.

### Model Highlights
- Each modality is treated as a **token**
- Tokens + `[CLS]` are processed by a Transformer encoder
- Optional modality-type embeddings
- End-to-end trainable fusion and classification

### Directory Overview
C_ViT_Fusion/
├─ src/
│ ├─ data/ # CSV loading & Dataset
│ ├─ models/ # ViT-based fusion network
│ ├─ utils/ # Metrics, plots, reproducibility
│ └─ train_fusion.py
├─ examples/
│ └─ demo_features/ # Synthetic / anonymized demo CSVs
└─ docs/
└─ feature_interface.md

### Demo Feature Files
A minimal set of **example feature CSVs** is provided for smoke testing:

examples/demo_features/
├─ labeRND-0-group.csv
├─ radiomics_selected.csv
├─ feat2d_selected.csv
├─ feat3d_selected.csv
└─ clinical.csv

These files:
- Do **not** contain patient-identifying information
- Are intended only to verify that the fusion pipeline runs correctly

---

## 🔁 Reproducibility Strategy

- **Module A**: Fully reproducible with user-provided NIfTI data
- **Module B**: Interface-level reproducibility (feature definitions & formats)
- **Module C**: Fully reproducible ViT-based fusion and training code

Researchers may substitute their own implementations for Module B as long as
the feature interface is respected.

---

## 📜 Notes on Data Privacy and Licensing

- No raw medical images are included in this repository.
- No proprietary source code is redistributed.
- Users are responsible for complying with local data protection regulations
  and third-party software licenses.

---

## 📧 Contact

For questions regarding methodology or reproduction, please contact the authors
through the corresponding publication.

---

**This repository is intended for academic research use only.**
