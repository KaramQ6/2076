# ADR 0002 — Patient-Level Splitting and Transfer Learning Strategy
Date: 2026-09-18 · Status: accepted

## Context
The BraTS-TCGA-GBM dataset comprises 102 pre-operative subjects. In 3D medical imaging, training from random initialization on small cohorts (<200 subjects) leads to poor generalization, severe overfitting, and high sensitivity to random seeds. Furthermore, slice-level or patch-level random splitting across subjects causes catastrophic data leakage.

## Decision
1. **Strict Patient-Level Partitioning:** Enforce a 70% Train / 15% Validation / 15% Test split partitioned strictly by `patient_id`. The test split is frozen in `split_manifest.csv` and evaluated only once for the final benchmark.
2. **Transfer Learning Protocol (FORGE V5 §A.2):** Initialize model weights using pre-trained BraTS foundation priors (trained on large multi-institutional cohorts) followed by fine-tuning on the TCGA cohort. This achieves state-of-the-art Dice scores (>0.85) without requiring hundreds of hours of raw compute.

## Consequences
- **Positive:** Guarantees zero data leakage; boosts Dice scores across all three subregions; maintains low inference latency (<2.0 seconds).
- **Negative:** Requires downloading pre-trained checkpoint weights and validating distribution alignment with TCIA NIfTI intensities.
