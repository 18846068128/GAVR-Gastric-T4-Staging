import argparse
import os
import random
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Optional: disable torch inductor if you had issues on your environment
os.environ.setdefault("TORCHINDUCTOR_DISABLE", "1")

from datasets.nifti_2d_slice_dataset import Nifti2DSliceSegDataset
from losses.dice import DiceLoss
from models.unet2d import UNet2D


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # deterministic can reduce performance but improves reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def match_image_mask_files(images_dir: str, masks_dir: str) -> Tuple[List[str], List[str]]:
    """
    Match images and masks by identical filename (e.g., case001.nii.gz).
    Only keep pairs that exist in both folders.
    """
    image_files = sorted(
        [os.path.join(images_dir, f) for f in os.listdir(images_dir) if f.endswith(".nii.gz")]
    )
    mask_files = []
    matched_images = []

    for img_path in image_files:
        fname = os.path.basename(img_path)
        mpath = os.path.join(masks_dir, fname)
        if os.path.exists(mpath):
            matched_images.append(img_path)
            mask_files.append(mpath)

    return matched_images, mask_files


def train_model_seg(
    model: nn.Module,
    dataset: torch.utils.data.Dataset,
    device: torch.device,
    num_epochs: int = 100,
    batch_size: int = 4,
    lr: float = 1e-4,
    num_workers: int = 0,
    save_path: str = "outputs/unet2d_seg.pth",
):
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    bce_logits = nn.BCEWithLogitsLoss()
    dice = DiceLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))
    model.to(device)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0

        for imgs, masks in loader:
            imgs = imgs.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                logits = model(imgs)
                loss = 0.5 * bce_logits(logits, masks) + 0.5 * dice(logits, masks)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += float(loss.item())

        avg_loss = total_loss / max(len(loader), 1)
        print(f"[Epoch {epoch + 1}/{num_epochs}] Loss: {avg_loss:.6f}")

    torch.save(model.state_dict(), save_path)
    print(f"✅ Model saved to: {save_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="2D U-Net training for ROI segmentation (NIfTI -> 2D slices)")
    parser.add_argument("--images_dir", type=str, required=True, help="Folder with image .nii.gz files")
    parser.add_argument("--masks_dir", type=str, required=True, help="Folder with mask .nii.gz files (same filenames)")
    parser.add_argument("--target_h", type=int, default=256)
    parser.add_argument("--target_w", type=int, default=256)
    parser.add_argument("--augment", action="store_true", help="Enable data augmentation (flip/rotate)")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--base_channels", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--save_path", type=str, default="outputs/unet2d_seg.pth")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    image_files, mask_files = match_image_mask_files(args.images_dir, args.masks_dir)
    print(f"✅ Matched cases: {len(image_files)}")

    if len(image_files) == 0:
        raise RuntimeError("No matched image-mask pairs found. Check filenames and folders.")

    dataset = Nifti2DSliceSegDataset(
        image_files,
        mask_files,
        target_size=(args.target_h, args.target_w),
        augment=args.augment,
    )

    model = UNet2D(in_channels=1, out_channels=1, base_channels=args.base_channels)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Current device:", device)

    train_model_seg(
        model=model,
        dataset=dataset,
        device=device,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        num_workers=args.num_workers,
        save_path=args.save_path,
    )


if __name__ == "__main__":
    main()
