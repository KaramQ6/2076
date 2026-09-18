# ADR 0001 — SegResNet 3D Architecture and Multi-Label Sigmoid Activation
Date: 2026-09-18 · Status: accepted

## Context
Glioma segmentation requires delineating three clinically distinct yet spatially overlapping anatomical compartments from 4 multimodal MRI sequences (`T1`, `T1ce`, `T2`, `FLAIR`):
1. Tumor Core (TC: Necrotic core + Enhancing tumor)
2. Whole Tumor (WT: Core + Edema + Enhancing tumor)
3. Enhancing Tumor (ET)

Because WT contains TC, and TC contains ET, the classes are non-mutually exclusive. Standard multi-class Softmax activation forces mutual exclusivity, breaking anatomical containment and penalizing boundary overlap.

## Decision
1. We choose `MONAI SegResNet 3D` as our primary architectural backbone due to its superior parameter efficiency and memory footprint on 8GB consumer GPUs compared to standard 3D U-Net.
2. We enforce `DiceCELoss(sigmoid=True, include_background=True, to_onehot_y=False)` across three independent binary sigmoid channels: Channel 0: TC, Channel 1: WT, Channel 2: ET.

## Consequences
- **Positive:** Enables natural learning of hierarchical overlapping compartments; eliminates softmax competition; reduces VRAM consumption during sliding window inference.
- **Negative:** Requires independent thresholding (default 0.5) per channel and careful post-processing to synthesize consolidated multi-value integer masks.
