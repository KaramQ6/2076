"""
Inference Engine for 3D Multimodal Glioma Segmentation
======================================================

Executes sliding-window inference with MONAI SegResNet, applies sigmoid multi-label
activation, saves 3D NIfTI segmentation masks, and exports clinical overlays.
"""

from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List
import os
import torch
import numpy as np
import nibabel as nib
from monai.networks.nets import SegResNet
from monai.inferers import sliding_window_inference
from monai.transforms import SpatialPad, SpatialCrop
from glioma_ai.inference.overlay_generator import (
    compute_tumor_volumes_cm3,
    generate_clinical_overlays,
)


class GliomaPredictor:
    """
    Inference wrapper for 3D Glioma segmentation.
    """

    def __init__(
        self,
        weights_path: Optional[str] = None,
        roi_size: Tuple[int, int, int] = (240, 240, 160),
        sw_overlap: float = 0.5,
        device: Optional[str] = None,
    ):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        if weights_path is None:
            pkg_dir = Path(__file__).resolve().parent.parent
            sota_path = pkg_dir / "weights" / "sota_segresnet_weights.pth"
            best_path = pkg_dir / "weights" / "best_segresnet_weights.pth"
            if sota_path.exists():
                weights_path = str(sota_path)
            elif best_path.exists():
                weights_path = str(best_path)

        self.roi_size = roi_size
        self.sw_overlap = sw_overlap

        # Default SOTA SegResNet configuration
        init_filters = 16
        blocks_down = [1, 2, 2, 4]
        blocks_up = [1, 1, 1]

        state_dict = None
        if weights_path and os.path.exists(weights_path):
            state_dict = torch.load(weights_path, map_location=self.device)
            if "convInit.weight" in state_dict:
                init_filters = state_dict["convInit.weight"].shape[0]

        # Initialize SegResNet architecture (4 MRI input channels -> 3 tumor output channels)
        self.model = SegResNet(
            spatial_dims=3,
            in_channels=4,
            out_channels=3,
            init_filters=init_filters,
            blocks_down=blocks_down,
            blocks_up=blocks_up,
            dropout_prob=0.0,
        ).to(self.device)

        if state_dict is not None:
            self.model.load_state_dict(state_dict)
            self.is_trained = True
        else:
            self.is_trained = False

        self.model.eval()

    def predict(
        self,
        sequence_paths: Dict[str, str],
        output_dir: str,
        case_id: str = "CASE_01",
    ) -> Dict[str, Any]:
        """
        Runs full 3D inference given paths to T1, T1ce, T2, FLAIR.
        
        Returns:
          Dict with:
            - status: "succeeded"
            - measurements: List[Dict] with volume in cm3
            - artifact_refs: List of created file paths (NIfTI mask, PNG overlays)
            - metadata: voxel spacing, volume shape, etc.
        """
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        # 1. Load the 4 MRI volumes using nibabel
        t1_nii = nib.load(sequence_paths["t1"])
        t1ce_nii = nib.load(sequence_paths["t1ce"])
        t2_nii = nib.load(sequence_paths["t2"])
        flair_nii = nib.load(sequence_paths["flair"])

        affine = t1ce_nii.affine
        voxel_spacing = t1ce_nii.header.get_zooms()[:3]

        arr_t1 = t1_nii.get_fdata(dtype=np.float32)
        arr_t1ce = t1ce_nii.get_fdata(dtype=np.float32)
        arr_t2 = t2_nii.get_fdata(dtype=np.float32)
        arr_flair = flair_nii.get_fdata(dtype=np.float32)

        # Channel-wise nonzero intensity normalization (Standard order: T1ce, T1, T2, FLAIR)
        channels = []
        for arr in [arr_t1ce, arr_t1, arr_t2, arr_flair]:
            mask = arr > 0
            if mask.any():
                mean = arr[mask].mean()
                std = arr[mask].std()
                norm = np.zeros_like(arr)
                norm[mask] = (arr[mask] - mean) / (std + 1e-8)
            else:
                norm = np.zeros_like(arr)
            channels.append(norm)

        # Stack into shape: (1, 4, D, H, W)
        stacked = np.stack(channels, axis=0)
        tensor_in = torch.from_numpy(stacked).unsqueeze(0).to(self.device)

        # 2. Sliding Window Inference with AMP (adaptive padding for 240x240x155 BraTS volumes)
        orig_shape = tuple(tensor_in.shape[2:])
        needs_pad = (orig_shape == (240, 240, 155))
        if needs_pad:
            padder = SpatialPad(spatial_size=(240, 240, 160))
            padded_in = padder(tensor_in.squeeze(0)).unsqueeze(0).to(self.device)
        else:
            padded_in = tensor_in

        with torch.no_grad():
            with torch.amp.autocast(device_type="cuda" if self.device.type == "cuda" else "cpu", enabled=self.device.type == "cuda"):
                raw_logits = sliding_window_inference(
                    inputs=padded_in,
                    roi_size=self.roi_size,
                    sw_batch_size=1,
                    predictor=self.model,
                    overlap=self.sw_overlap,
                    mode="gaussian",
                )
                padded_probs = torch.sigmoid(raw_logits).squeeze(0)

        if needs_pad:
            cropper = SpatialCrop(roi_size=(240, 240, 155), roi_center=(120, 120, 80))
            probs = cropper(padded_probs).cpu().numpy()
        else:
            probs = padded_probs.cpu().numpy()

        # 3. Threshold at 0.5 for multi-label binary mask (3, D, H, W)
        pred_mask_3ch = (probs > 0.5).astype(np.uint8)

        # 4. Synthesize integer 3D label map for standard NIfTI export:
        # 1 = Necrotic Core (TC but not ET)
        # 2 = Edema (WT but not TC)
        # 4 = Enhancing Tumor (ET)
        label_map = np.zeros(pred_mask_3ch.shape[1:], dtype=np.int16)
        tc = pred_mask_3ch[0] == 1
        wt = pred_mask_3ch[1] == 1
        et = pred_mask_3ch[2] == 1

        label_map[wt] = 2
        label_map[tc] = 1
        label_map[et] = 4

        # Save NIfTI mask
        seg_filename = f"{case_id}_segmentation.nii.gz"
        seg_filepath = out_dir / seg_filename
        seg_nii = nib.Nifti1Image(label_map, affine)
        nib.save(seg_nii, str(seg_filepath))

        # 5. Compute quantitative volumetrics in cm3
        volumes = compute_tumor_volumes_cm3(pred_mask_3ch, voxel_spacing=voxel_spacing)

        # 6. Generate 3-plane overlays
        overlay_files = generate_clinical_overlays(
            mri_volume=arr_t1ce, # Background image: post-contrast T1
            pred_mask_3ch=pred_mask_3ch,
            output_dir=str(out_dir),
            prefix=f"{case_id}_overlay",
        )

        artifact_refs = [str(seg_filepath)] + list(overlay_files.values())

        return {
            "status": "succeeded",
            "measurements": [
                {"name": "whole_tumor_volume", "value": volumes["whole_tumor_volume"], "unit": "cm3"},
                {"name": "tumor_core_volume", "value": volumes["tumor_core_volume"], "unit": "cm3"},
                {"name": "enhancing_tumor_volume", "value": volumes["enhancing_tumor_volume"], "unit": "cm3"},
            ],
            "artifact_refs": artifact_refs,
            "voxel_spacing": list(voxel_spacing),
            "volume_shape": list(arr_t1ce.shape),
        }
