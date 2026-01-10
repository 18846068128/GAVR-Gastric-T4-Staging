import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def save_png_overlay_prob_mask(
    orig_slice: np.ndarray,
    prob_map: np.ndarray,
    mask_bin: np.ndarray,
    save_path_overlay: str,
    save_path_prob: str,
    save_path_mask: str,
):
    """
    Save:
      1) overlay: gray image + red mask (alpha)
      2) prob: gray image + jet heatmap + colorbar
      3) mask: single-channel binary PNG

    orig_slice: 2D float array in [0,1]
    prob_map:  2D float array in [0,1]
    mask_bin:  2D uint8 array (0/1)
    """
    os.makedirs(os.path.dirname(save_path_overlay), exist_ok=True)

    orig_uint8 = (np.clip(orig_slice, 0, 1) * 255).astype(np.uint8)
    prob_uint8 = (np.clip(prob_map, 0, 1) * 255).astype(np.uint8)
    mask_uint8 = (mask_bin.astype(np.uint8) * 255)

    # 1) Overlay
    fig, ax = plt.subplots(figsize=(6, 6), dpi=150)
    ax.imshow(orig_uint8, cmap="gray")
    ax.imshow(mask_uint8, cmap="Reds", alpha=0.4)
    ax.axis("off")
    fig.savefig(save_path_overlay, bbox_inches="tight", pad_inches=0)
    plt.close(fig)

    # 2) Prob heatmap
    fig2, ax2 = plt.subplots(figsize=(6, 6), dpi=150)
    ax2.imshow(orig_uint8, cmap="gray")
    im = ax2.imshow(prob_uint8, cmap="jet", alpha=0.5)
    ax2.axis("off")
    fig2.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
    fig2.savefig(save_path_prob, bbox_inches="tight", pad_inches=0)
    plt.close(fig2)

    # 3) Binary mask
    Image.fromarray(mask_uint8).save(save_path_mask)
