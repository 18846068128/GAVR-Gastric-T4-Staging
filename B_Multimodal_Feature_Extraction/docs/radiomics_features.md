# Radiomics Feature Extraction

Radiomics features are extracted using the open-source **PyRadiomics** library.

---

## ROI Definition

- Tumor ROIs are obtained from Module A (2D U-Net segmentation).
- Binary masks are used to define the region of interest.

---

## Feature Categories

Radiomics features include (but are not limited to):

- First-order statistics
- Shape features
- GLCM (Gray-Level Co-occurrence Matrix)
- GLRLM (Gray-Level Run Length Matrix)
- GLSZM (Gray-Level Size Zone Matrix)
- GLDM (Gray-Level Dependence Matrix)
- NGTDM (Neighboring Gray Tone Difference Matrix)

---

## Output Format

- One row per case
- One column per radiomic feature
- CSV format with an `ID` column

---

## Reproducibility

Any PyRadiomics-based implementation producing equivalent features
can be used to reproduce this modality.
