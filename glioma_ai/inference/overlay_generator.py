"""
Clinical Visualization Overlays & Quantitative Tumor Volumetrics
================================================================

Generates reviewable multi-planar transparent overlays (Axial, Coronal, Sagittal)
and computes physical tumor volumes in cm³ for clinician second-reader review.
"""

from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import nibabel as nib
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def compute_tumor_volumes_cm3(
    pred_mask_3ch: np.ndarray,
    voxel_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> Dict[str, float]:
    """
    Computes volumetric tumor measurements in cubic centimeters (cm³).
    
    Channels:
      Channel 0: TC (Tumor Core)
      Channel 1: WT (Whole Tumor)
      Channel 2: ET (Enhancing Tumor)
    
    Formula:
      Volume (cm³) = (voxel_count * dx * dy * dz) / 1000.0
    """
    voxel_vol_mm3 = float(voxel_spacing[0] * voxel_spacing[1] * voxel_spacing[2])
    voxel_vol_cm3 = voxel_vol_mm3 / 1000.0

    tc_count = int(np.sum(pred_mask_3ch[0] > 0.5))
    wt_count = int(np.sum(pred_mask_3ch[1] > 0.5))
    et_count = int(np.sum(pred_mask_3ch[2] > 0.5))

    return {
        "whole_tumor_volume": round(wt_count * voxel_vol_cm3, 2),
        "tumor_core_volume": round(tc_count * voxel_vol_cm3, 2),
        "enhancing_tumor_volume": round(et_count * voxel_vol_cm3, 2),
    }


def generate_clinical_overlays(
    mri_volume: np.ndarray,
    pred_mask_3ch: np.ndarray,
    output_dir: str,
    prefix: str = "overlay",
) -> Dict[str, str]:
    """
    Renders anatomical overlays across Axial, Coronal, and Sagittal planes
    at the slice with maximum tumor burden.
    
    Color Convention:
      - Whole Tumor (edema periphery): Green (alpha=0.4)
      - Tumor Core (necrotic): Blue (alpha=0.5)
      - Enhancing Tumor: Red (alpha=0.7)
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Normalize MRI background to 0..255 for clean visual contrast
    mri_norm = mri_volume.astype(np.float32)
    p1, p99 = np.percentile(mri_norm, [1, 99])
    if p99 > p1:
        mri_norm = np.clip((mri_norm - p1) / (p99 - p1), 0, 1)
    else:
        mri_norm = np.zeros_like(mri_norm)

    wt_mask = (pred_mask_3ch[1] > 0.5).astype(np.uint8)
    tc_mask = (pred_mask_3ch[0] > 0.5).astype(np.uint8)
    et_mask = (pred_mask_3ch[2] > 0.5).astype(np.uint8)

    # Find peak tumor slice along each axis (depth/axial, height/coronal, width/sagittal)
    # If no tumor, select center slice
    if np.sum(wt_mask) > 0:
        ax_idx = int(np.argmax(np.sum(wt_mask, axis=(1, 2))))
        cor_idx = int(np.argmax(np.sum(wt_mask, axis=(0, 2))))
        sag_idx = int(np.argmax(np.sum(wt_mask, axis=(0, 1))))
    else:
        ax_idx = mri_volume.shape[0] // 2
        cor_idx = mri_volume.shape[1] // 2
        sag_idx = mri_volume.shape[2] // 2

    views = {
        "axial": (mri_norm[ax_idx, :, :], wt_mask[ax_idx, :, :], tc_mask[ax_idx, :, :], et_mask[ax_idx, :, :], ax_idx),
        "coronal": (mri_norm[:, cor_idx, :], wt_mask[:, cor_idx, :], tc_mask[:, cor_idx, :], et_mask[:, cor_idx, :], cor_idx),
        "sagittal": (mri_norm[:, :, sag_idx], wt_mask[:, :, sag_idx], tc_mask[:, :, sag_idx], et_mask[:, :, sag_idx], sag_idx),
    }

    generated_files = {}

    for view_name, (bg_slice, wt_sl, tc_sl, et_sl, slice_no) in views.items():
        fig, ax = plt.subplots(figsize=(6, 6), dpi=150)
        ax.imshow(bg_slice, cmap="gray", origin="lower")

        # Build RGB overlay
        h, w = bg_slice.shape
        rgba = np.zeros((h, w, 4), dtype=np.float32)

        # Whole Tumor (Green)
        rgba[wt_sl == 1] = [0.0, 0.9, 0.2, 0.35]
        # Tumor Core (Blue/Orange distinction)
        rgba[tc_sl == 1] = [0.1, 0.4, 0.95, 0.55]
        # Enhancing Tumor (Bright Red)
        rgba[et_sl == 1] = [0.95, 0.1, 0.1, 0.75]

        ax.imshow(rgba, origin="lower")
        ax.axis("off")
        ax.set_title(f"Glioma Subregions ({view_name.capitalize()} Slice #{slice_no})", color="white", fontsize=11, pad=8)

        # Legend
        patches = [
            mpatches.Patch(color=(0.0, 0.9, 0.2, 0.6), label="Whole Tumor (WT)"),
            mpatches.Patch(color=(0.1, 0.4, 0.95, 0.7), label="Tumor Core (TC)"),
            mpatches.Patch(color=(0.95, 0.1, 0.1, 0.85), label="Enhancing Tumor (ET)"),
        ]
        legend = ax.legend(handles=patches, loc="lower right", facecolor="#111827", edgecolor="#374151", fontsize=8)
        for text in legend.get_texts():
            text.set_color("white")

        fig.patch.set_facecolor("#0B0F19")
        file_path = out_path / f"{prefix}_{view_name}.png"
        plt.tight_layout()
        plt.savefig(str(file_path), facecolor=fig.get_facecolor(), bbox_inches="tight")
        plt.close(fig)

        generated_files[f"overlay_{view_name}"] = str(file_path)

    return generated_files
