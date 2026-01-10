# Data Format and Organization (Module A: ROI Segmentation)

This document describes the required data format and directory structure
for **Module A: ROI automatic segmentation using 2D U-Net**.

---

## 1. Directory Structure

The dataset should be organized into two folders: `images/` and `masks/`.
Each image volume must have a corresponding mask volume with the **same filename**.

data/
├─ images/
│ ├─ case001.nii.gz
│ ├─ case002.nii.gz
│ └─ ...
├─ masks/
│ ├─ case001.nii.gz
│ ├─ case002.nii.gz
│ └─ ...

Image–mask pairs are matched strictly by filename.

---

## 2. File Format

- File format: **NIfTI** (`.nii.gz`)
- Each file must be a **3D volume**
- Expected array shape when loaded by `nibabel`:

(H, W, D)

where:
- `H`: image height
- `W`: image width
- `D`: number of slices (axial direction)

The third dimension (`D`) is treated as the slice index during training and inference.

---

## 3. Image Intensity Processing

- No global intensity normalization is required before input.
- During training and inference:
  - Each **2D slice** is normalized independently using min–max normalization:
    ```
    I_norm = (I - min(I)) / (max(I) - min(I))
    ```
- If a slice has near-constant intensity, it is mapped to zeros.

---

## 4. Mask Definition

- Mask volumes must have the same shape as the corresponding image volumes.
- Mask values are binarized internally:
  - All voxels with value `> 0` are treated as foreground (ROI)
  - Voxels with value `== 0` are treated as background
- Multi-class labels (if present) will be collapsed into a single foreground class.

---

## 5. Preprocessing Assumptions

- Input volumes may be:
  - cropped to the tumor region, or
  - full field-of-view volumes
- No assumption is made on voxel spacing or orientation;
  all processing is performed slice-wise in image space.
- If voxel spacing, orientation, or windowing differs across datasets,
  users should harmonize them **before** training for best performance.

---

## 6. Slice-Based Training Strategy

- Each 3D volume is decomposed into multiple 2D slices along the axial axis.
- All slices are used during training, including slices without foreground,
  which helps reduce false positives.
- Data augmentation is applied in 2D slice space.

---

## 7. Example

Example pair:

images/case012.nii.gz -> shape (512, 512, 78)
masks/case012.nii.gz -> shape (512, 512, 78)

Each slice `z` is processed independently as a `(1, H, W)` input to the 2D U-Net.

---

## 8. Notes on Data Privacy

- This repository does **not** include any raw medical image data.
- Users must ensure that all data used complies with local regulations
  and institutional review board (IRB) requirements.
