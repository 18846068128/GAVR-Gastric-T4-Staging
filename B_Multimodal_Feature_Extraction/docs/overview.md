# Module B: Methodological Overview

Module B corresponds to the **second-level cognitive architecture** in the proposed
framework. It bridges low-level ROI perception (Module A) and high-level
multimodal reasoning (Module C).

The objective of Module B is to convert heterogeneous information sources into
compact, discriminative, and complementary feature representations.

---

## Input

- Segmented tumor regions from Module A
- Original CT volumes
- Associated clinical records

---

## Output

For each patient/case, Module B outputs a set of feature vectors:

| Modality | Description | Output format |
|--------|------------|---------------|
| Radiomics | Handcrafted texture, shape, and intensity descriptors | CSV |
| 2D CNN | High-level semantic features from axial tumor slices | CSV |
| 3D CNN | Volumetric spatial features from tumor ROI | CSV |
| Clinical | Structured clinical variables | CSV |

---

## Design Philosophy

- **Complementarity**: Radiomics captures low-level patterns, CNNs capture
  high-level semantics, and clinical variables provide contextual priors.
- **Modularity**: Each modality is extracted independently and can be replaced
  by functionally equivalent methods.
- **Interface-driven**: Downstream fusion relies on standardized feature tables,
  not on specific extraction implementations.
