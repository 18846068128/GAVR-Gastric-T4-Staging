# Deep Learning–Based Image Features

Deep image features are extracted using convolutional neural networks applied to
the segmented tumor regions.

---

## 2D CNN Features (DenseNet121)

- Input: 2D tumor slices
- Backbone: DenseNet121
- Pretraining: ImageNet (or equivalent)
- Feature extraction: Global pooling layer embeddings
- Output: One fixed-length vector per case

---

## 3D CNN Features (ShuffleNet3D)

- Input: 3D tumor volume
- Backbone: ShuffleNet3D
- Feature extraction: Final convolutional embedding
- Output: One fixed-length vector per case

---

## Implementation Note

The CNN feature extraction pipelines are implemented using
**vendor-provided (onekey) code** and are not redistributed in this repository.

However, the output feature representations follow a standardized CSV format
and can be reproduced using alternative CNN implementations.
