# Feature Interface Specification (Module B → Module C)

This document specifies the **feature-level interface** between:
- **Module B**: Multi-modal feature extraction (external / partially proprietary), and
- **Module C**: ViT-based fusion for classification.

Module B can be treated as a black box. As long as it produces the feature tables
described here, Module C can be trained and evaluated reproducibly.

---

## 1. Overview

Module C expects four feature modalities per patient/case:

1) **Radiomics features** (PyRadiomics)
2) **2D CNN features** (e.g., DenseNet121 embedding)
3) **3D CNN features** (e.g., ShuffleNet3D embedding)
4) **Clinical variables** (tabular clinical features)

All modalities must be aligned by a unique **ID**.

---

## 2. Required Input Files

Module C consumes **five CSV files**:

### 2.1 Label File
`label.csv` must contain at least these columns:

| Column | Type | Description |
|---|---|---|
| ID | string | unique case identifier |
| group | string | split indicator (e.g., train/test) |
| label | int | class label index (e.g., 0/1) |

Example:

```csv
ID,group,label
case001,train,0
case002,train,1
case101,test,0
2.2 Feature Tables (4 modalities)

Each feature table must contain an ID column and feature columns:

radiomics.csv

feat_2d.csv

feat_3d.csv

clinical.csv

All of them follow the same structure:
| Column | Type   | Description                            |
| ------ | ------ | -------------------------------------- |
| ID     | string | unique case identifier                 |
| f1..fK | float  | feature values (K depends on modality) |
Example:
ID,f0,f1,f2,f3
case001,0.12,-1.03,3.20,0.00
case002,0.10,-0.88,2.91,0.10
Important:

Feature columns must be numeric (float/int).

Missing values should be handled before training (e.g., imputation).

The four feature tables must contain the same set of IDs as label.csv
for the corresponding split.

3. Alignment Rule

Module C aligns all modalities by matching ID:

For each case ID in label.csv, it retrieves the row with the same ID in each feature table.

If an ID is missing in any modality table, that case cannot be used unless handled explicitly.

4. Modality Dimensions (Dynamically Inferred)

Module C does not require hard-coded modality dimensions.
Dimensions are inferred from the CSV column counts:

Radiomics dimension: R = radiomics.csv columns - 1

2D feature dimension: D2 = feat_2d.csv columns - 1

3D feature dimension: D3 = feat_3d.csv columns - 1

Clinical dimension: C = clinical.csv columns - 1

These must be consistent across train/test for each modality.

5. Standardization (Recommended)

To reduce scale differences, standardization is recommended per modality:

Fit StandardScaler on training split only.

Apply the same transformation to test split.

This repository’s reference implementation uses per-modality StandardScaler.

6. Notes on External/Proprietary Feature Extraction

Module B may include components that are external or proprietary (e.g., vendor-provided models).
This repository does not distribute proprietary code.

However, any external implementation is acceptable as long as it outputs:

CSV files in the formats above,

aligned by ID,

and containing numeric feature columns.

7. Minimal Example Package

For demonstration and smoke testing, this repository may include a small synthetic/demo feature package:

examples/demo_features/label.csv

examples/demo_features/radiomics.csv

examples/demo_features/feat_2d.csv

examples/demo_features/feat_3d.csv

examples/demo_features/clinical.csv

These demo files do not contain patient-identifying information and are intended only
to verify the pipeline runs end-to-end.

---
