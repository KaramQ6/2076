"""
Unit tests for transforms and label harmonization
=================================================
Validates native BraTS (0, 1, 2, 4) and standardized channel output order:
  - Channel 0: TC (Necrotic Core + Enhancing)
  - Channel 1: WT (Whole Tumor: Core + Edema + Enhancing)
  - Channel 2: ET (Enhancing Tumor only)
"""

import torch
import numpy as np
import pytest
from monai.transforms import ConvertToMultiChannelBasedOnBratsClassesd
from glioma_ai.train.transforms import StandardizeBraTSLabelsd


def test_standardize_brats_labels():
    """Verify that Label 3 (if present without 4) is standardized to 4."""
    labels_with_3 = torch.tensor([0, 1, 2, 3], dtype=torch.int64)
    data = {"label": labels_with_3}
    transform = StandardizeBraTSLabelsd(keys=["label"])
    res = transform(data)
    
    expected = torch.tensor([0, 1, 2, 4], dtype=torch.int64)
    assert torch.equal(res["label"], expected), f"Expected {expected}, got {res['label']}"


def test_raw_brats_labels_channel_order():
    """
    Verify the exact channel mapping from MONAI's ConvertToMultiChannelBasedOnBratsClassesd
    directly on raw BraTS-TCGA-GBM labels (0, 1, 2, 4):
    Expected:
      Channel 0: TC (Necrotic Core + Enhancing Tumor)
      Channel 1: WT (Whole Tumor: Core + Edema + Enhancing)
      Channel 2: ET (Enhancing Tumor only)
    """
    # Create synthetic volume with 1 voxel of each label:
    # voxel (0,0,0) = 0 (background)
    # voxel (0,0,1) = 1 (necrotic core)
    # voxel (0,0,2) = 2 (edema)
    # voxel (0,0,3) = 4 (enhancing tumor)
    vol = torch.zeros((1, 1, 4), dtype=torch.int64)
    vol[0, 0, 1] = 1
    vol[0, 0, 2] = 2
    vol[0, 0, 3] = 4

    data = {"label": vol}
    converter = ConvertToMultiChannelBasedOnBratsClassesd(keys=["label"], et_label=4)
    res = converter(data)
    multi_chan = res["label"] # Shape: (3, 1, 1, 4)

    assert multi_chan.shape[0] == 3, f"Expected 3 channels, got {multi_chan.shape[0]}"

    # Check Channel 0: TC (active for label 1 and label 4)
    assert multi_chan[0, 0, 0, 0] == 0 # background
    assert multi_chan[0, 0, 0, 1] == 1 # necrotic -> TC active
    assert multi_chan[0, 0, 0, 2] == 0 # edema -> TC inactive
    assert multi_chan[0, 0, 0, 3] == 1 # enhancing -> TC active

    # Check Channel 1: WT (Whole Tumor: active for label 1, 2, 4)
    assert multi_chan[1, 0, 0, 0] == 0 # background
    assert multi_chan[1, 0, 0, 1] == 1 # necrotic -> WT active
    assert multi_chan[1, 0, 0, 2] == 1 # edema -> WT active
    assert multi_chan[1, 0, 0, 3] == 1 # enhancing -> WT active

    # Check Channel 2: ET (Enhancing Tumor: active ONLY for label 4)
    assert multi_chan[2, 0, 0, 0] == 0 # background
    assert multi_chan[2, 0, 0, 1] == 0 # necrotic -> ET inactive
    assert multi_chan[2, 0, 0, 2] == 0 # edema -> ET inactive
    assert multi_chan[2, 0, 0, 3] == 1 # enhancing -> ET active
