import random
from typing import List, Tuple

import nibabel as nib
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import functional as TF


class Nifti2DSliceSegDataset(Dataset):
    """
    3D NIfTI (H, W, D) -> 2D slices along axis=2 (z).
    - Load all volumes into memory once (fast, but RAM heavy).
    - Normalize each slice to [0,1] using slice min/max.
    - Binarize mask: mask > 0 -> 1.0
    - Augmentations: random horizontal flip, random rotate {0,90,180,270}
    - Resize to target_size and convert to tensor (C,H,W), C=1
    """

    def __init__(
        self,
        nifti_image_files: List[str],
        nifti_mask_files: List[str],
        target_size: Tuple[int, int] = (256, 256),
        augment: bool = True,
    ):
        if len(nifti_image_files) != len(nifti_mask_files):
            raise ValueError(
                f"Image/mask count mismatch: {len(nifti_image_files)} vs {len(nifti_mask_files)}"
            )

        self.image_files = nifti_image_files
        self.mask_files = nifti_mask_files
        self.target_size = target_size
        self.augment = augment

        # Load volumes once
        self.images = [nib.load(p).get_fdata() for p in self.image_files]
        self.masks = [nib.load(p).get_fdata() for p in self.mask_files]

        # Build slice index list: (case_idx, z)
        self.slices = []
        for case_idx, img_np in enumerate(self.images):
            if img_np.ndim != 3:
                raise ValueError(f"Expected 3D NIfTI, got shape {img_np.shape} for {self.image_files[case_idx]}")
            _, _, D = img_np.shape
            for z in range(D):
                self.slices.append((case_idx, z))

        self.resize = transforms.Resize(self.target_size, antialias=True)
        self.to_tensor = transforms.ToTensor()

    def __len__(self) -> int:
        return len(self.slices)

    def __getitem__(self, index: int):
        case_idx, z = self.slices[index]

        img_slice = self.images[case_idx][:, :, z]
        mask_slice = self.masks[case_idx][:, :, z]

        # Normalize image slice to [0,1]
        img_min = float(np.min(img_slice))
        img_max = float(np.max(img_slice))
        img_slice = (img_slice - img_min) / (img_max - img_min + 1e-8)

        # Binarize mask
        mask_slice = (mask_slice > 0).astype(np.float32)

        # Convert to PIL (uint8)
        img_pil = Image.fromarray((img_slice * 255).astype(np.uint8))
        mask_pil = Image.fromarray((mask_slice * 255).astype(np.uint8))

        # Augmentations (keep image/mask aligned)
        if self.augment and random.random() > 0.5:
            img_pil = TF.hflip(img_pil)
            mask_pil = TF.hflip(mask_pil)

        if self.augment and random.random() > 0.5:
            angle = random.choice([0, 90, 180, 270])
            img_pil = TF.rotate(img_pil, angle)
            mask_pil = TF.rotate(mask_pil, angle)

        # Resize
        img_resized = self.resize(img_pil)
        mask_resized = self.resize(mask_pil)

        # To tensor: (1,H,W) in [0,1]
        img_tensor = self.to_tensor(img_resized)
        mask_tensor = self.to_tensor(mask_resized)

        # Ensure float32
        img_tensor = img_tensor.float()
        mask_tensor = mask_tensor.float()

        return img_tensor, mask_tensor
