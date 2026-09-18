"""
Medical Image Transformations & Label Harmonization
===================================================

Handles multimodal MRI sequence stacking (T1, T1ce, T2, FLAIR) and implements
robust multi-channel conversion into standard BraTS composite regions:
  - Channel 0: TC (Tumor Core = Necrotic + Enhancing)
  - Channel 1: WT (Whole Tumor = Necrotic + Edema + Enhancing)
  - Channel 2: ET (Enhancing Tumor)
"""

from typing import Dict, Any, List, Optional
import torch
import numpy as np
from monai.transforms import (
    MapTransform,
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    Spacingd,
    NormalizeIntensityd,
    RandSpatialCropd,
    RandFlipd,
    ConvertToMultiChannelBasedOnBratsClassesd,
    CastToTyped,
)


class StandardizeBraTSLabelsd(MapTransform):
    """
    Standardizes label representations across BraTS revisions.
    
    Why: Raw BraTS-TCGA-GBM labels are {0: Background, 1: Necrotic/Non-enhancing Core,
    2: Edema, 4: Enhancing Tumor}.
    If labels contain 3 instead of 4 (as in BraTS 2023), this transform harmonizes them
    to standard BraTS-TCGA-GBM label 4 so the downstream pipeline remains uniform.
    """

    def __init__(self, keys: List[str], allow_missing_keys: bool = False):
        super().__init__(keys, allow_missing_keys)

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        d = dict(data)
        for key in self.key_iterator(d):
            val = d[key]
            # If label 3 is present but not 4, standardize 3 -> 4 for consistency
            if isinstance(val, torch.Tensor):
                if (val == 3).any() and not (val == 4).any():
                    val[val == 3] = 4
            elif isinstance(val, np.ndarray):
                if np.any(val == 3) and not np.any(val == 4):
                    val[val == 3] = 4
            d[key] = val
        return d


def get_train_transforms(roi_size: tuple = (96, 96, 96)) -> Compose:
    """
    Returns training transformation pipeline with data augmentation.
    
    Channels produced in label:
      Channel 0: TC (Tumor Core: necrotic + enhancing)
      Channel 1: WT (Whole Tumor: core + edema + enhancing)
      Channel 2: ET (Enhancing Tumor)
    """
    return Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            StandardizeBraTSLabelsd(keys=["label"]),
            ConvertToMultiChannelBasedOnBratsClassesd(keys=["label"], et_label=4),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(
                keys=["image", "label"],
                pixdim=(1.0, 1.0, 1.0),
                mode=("bilinear", "nearest"),
            ),
            RandSpatialCropd(
                keys=["image", "label"],
                roi_size=roi_size,
                random_size=False,
            ),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            CastToTyped(keys=["image", "label"], dtype=[torch.float32, torch.float32]),
        ]
    )


def get_val_transforms() -> Compose:
    """
    Returns validation/evaluation transformation pipeline without stochastic augmentations.
    """
    return Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            StandardizeBraTSLabelsd(keys=["label"]),
            ConvertToMultiChannelBasedOnBratsClassesd(keys=["label"], et_label=4),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(
                keys=["image", "label"],
                pixdim=(1.0, 1.0, 1.0),
                mode=("bilinear", "nearest"),
            ),
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            CastToTyped(keys=["image", "label"], dtype=[torch.float32, torch.float32]),
        ]
    )


def get_inference_transforms() -> Compose:
    """
    Returns inference-time transformation pipeline for raw multi-sequence inputs (no ground truth).
    """
    return Compose(
        [
            LoadImaged(keys=["image"]),
            EnsureChannelFirstd(keys=["image"]),
            Orientationd(keys=["image"], axcodes="RAS"),
            Spacingd(
                keys=["image"],
                pixdim=(1.0, 1.0, 1.0),
                mode="bilinear",
            ),
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            CastToTyped(keys=["image"], dtype=torch.float32),
        ]
    )
