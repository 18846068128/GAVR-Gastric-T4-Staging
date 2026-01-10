import argparse
import os
from typing import Optional, Tuple, List

import nibabel as nib
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from models.unet2d import UNet2D
from utils.viz import save_png_overlay_prob_mask


def load_nifti(path: str):
    nii = nib.load(path)
    return nii.get_fdata(), nii.affine


def normalize_slice_minmax(slice_img: np.ndarray) -> np.ndarray:
    """Normalize 2D slice to [0,1] with per-slice min/max."""
    slice_img = slice_img.astype(np.float32)
    mn = float(slice_img.min())
    mx = float(slice_img.max())
    if mx - mn < 1e-6:
        return np.zeros_like(slice_img, dtype=np.float32)
    return (slice_img - mn) / (mx - mn)


def infer_and_save_slices_png(
    model: torch.nn.Module,
    nifti_path: str,
    device: torch.device,
    save_root: str,
    threshold: float = 0.5,
    model_input_size: Optional[Tuple[int, int]] = (256, 256),
) -> List[int]:
    """
    Run inference per-slice and save:
      - overlay PNG
      - probability heatmap PNG
      - binary mask PNG

    model_input_size:
      - If not None, resize input slice -> model_input_size before inference,
        then resize prob map back to original size for saving.
      - Use the same size as training resize (e.g., 256x256).
    """
    model.eval()

    img_np, _ = load_nifti(nifti_path)
    if img_np.ndim != 3:
        raise ValueError(f"Expected 3D NIfTI, got shape {img_np.shape}: {nifti_path}")

    H, W, D = img_np.shape

    patient_name = os.path.basename(nifti_path).replace(".nii.gz", "")
    out_dir = os.path.join(save_root, patient_name)
    overlay_dir = os.path.join(out_dir, "overlays")
    prob_dir = os.path.join(out_dir, "prob")
    mask_dir = os.path.join(out_dir, "mask")
    os.makedirs(overlay_dir, exist_ok=True)
    os.makedirs(prob_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)

    # Transforms
    to_tensor = transforms.ToTensor()
    resize_to_model = transforms.Resize(model_input_size, antialias=True) if model_input_size else None

    nonzero_slices = []

    for z in range(D):
        slice_img = img_np[:, :, z]
        slice_norm = normalize_slice_minmax(slice_img)

        # Prepare tensor (1,1,h,w)
        if resize_to_model is not None:
            pil = Image.fromarray((slice_norm * 255).astype(np.uint8))
            pil_in = resize_to_model(pil)
            x = to_tensor(pil_in).unsqueeze(0)  # (1,1,h,w)
        else:
            x = torch.from_numpy(slice_norm).unsqueeze(0).unsqueeze(0)  # (1,1,H,W)

        x = x.to(device).float()

        with torch.no_grad():
            logits = model(x)  # (1,1,h,w) logits
            prob = torch.sigmoid(logits).squeeze().cpu().numpy().astype(np.float32)  # (h,w)

        # Resize prob back to original size for saving
        if resize_to_model is not None:
            prob_pil = Image.fromarray((np.clip(prob, 0, 1) * 255).astype(np.uint8))
            prob_resized = np.array(
                prob_pil.resize((W, H), resample=Image.BILINEAR)
            ).astype(np.float32) / 255.0
        else:
            prob_resized = prob

        mask_bin = (prob_resized > threshold).astype(np.uint8)
        if mask_bin.sum() > 0:
            nonzero_slices.append(z)

        overlay_path = os.path.join(overlay_dir, f"slice_{z:03d}_overlay.png")
        prob_path = os.path.join(prob_dir, f"slice_{z:03d}_prob.png")
        mask_path = os.path.join(mask_dir, f"slice_{z:03d}_mask.png")

        save_png_overlay_prob_mask(
            orig_slice=slice_norm,
            prob_map=prob_resized,
            mask_bin=mask_bin,
            save_path_overlay=overlay_path,
            save_path_prob=prob_path,
            save_path_mask=mask_path,
        )

        # basic logging
        if z % max(1, D // 6) == 0:
            print(
                f"{patient_name}: slice {z+1}/{D}, mean_prob={prob_resized.mean():.5f}, mask_pixels={mask_bin.sum()}"
            )

    print(f"✅ done {patient_name}. nonzero slices: {len(nonzero_slices)}/{D}")
    return nonzero_slices


def parse_args():
    p = argparse.ArgumentParser("2D U-Net inference (save overlay/prob/mask PNG per slice)")
    p.add_argument("--weights", type=str, required=True, help="Path to trained .pth weights")
    p.add_argument("--test_dir", type=str, required=True, help="Folder with test .nii.gz")
    p.add_argument("--save_root", type=str, default="results_png", help="Output root dir")
    p.add_argument("--threshold", type=float, default=0.5, help="Binarization threshold")
    p.add_argument("--input_h", type=int, default=256, help="Model input height (set 0 to disable resize)")
    p.add_argument("--input_w", type=int, default=256, help="Model input width (set 0 to disable resize)")
    p.add_argument("--base_channels", type=int, default=32, help="UNet base channels (must match training)")
    return p.parse_args()


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    model = UNet2D(in_channels=1, out_channels=1, base_channels=args.base_channels)
    state = torch.load(args.weights, map_location=device)
    model.load_state_dict(state)
    model.to(device).float()
    model.eval()

    model_input_size = None
    if args.input_h > 0 and args.input_w > 0:
        model_input_size = (args.input_h, args.input_w)

    os.makedirs(args.save_root, exist_ok=True)

    test_files = [
        os.path.join(args.test_dir, f)
        for f in os.listdir(args.test_dir)
        if f.endswith(".nii.gz")
    ]
    test_files.sort()

    if len(test_files) == 0:
        raise RuntimeError(f"No .nii.gz found in {args.test_dir}")

    for path in test_files:
        print("Processing:", path)
        nonzero = infer_and_save_slices_png(
            model=model,
            nifti_path=path,
            device=device,
            save_root=args.save_root,
            threshold=args.threshold,
            model_input_size=model_input_size,
        )
        if len(nonzero) == 0:
            print(f"⚠️ WARNING: empty mask for {os.path.basename(path)}")
        else:
            print(f"✅ Non-empty slices: {len(nonzero)}")

    print("All done.")


if __name__ == "__main__":
    main()
