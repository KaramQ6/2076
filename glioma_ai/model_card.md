# Model Card: Glioma SegResNet 3D

## 1. Model Details
- **Architecture:** MONAI SegResNet 3D (Residual Encoder-Decoder with deep supervision capability)
- **Model Version:** 1.0.0
- **Primary Task:** 3D Multimodal Brain MRI Glioma Subregion Segmentation
- **Framework:** PyTorch 2.7.0+cu126, MONAI 1.6.0
- **Spatial Strategy:** Sliding Window Inference (`roi_size=[96, 96, 96]`, `overlap=0.5`, `mode="gaussian"`)
- **Optimization Objective:** `DiceCELoss(sigmoid=True, include_background=True, to_onehot_y=False)`

---

## 2. Intended Clinical Use
- **Role:** AI Second Reader finding support for neuroradiologists and multidisciplinary tumor boards.
- **Workflow:**
  1. Multimodal sequences (`T1`, `T1ce`, `T2`, `FLAIR`) are ingested and verified for spatial co-registration.
  2. The inference engine predicts 3 anatomical subregions:
     - **Whole Tumor (WT):** Necrotic core + Active enhancing tissue + Peritumoral edema.
     - **Tumor Core (TC):** Necrotic core + Active enhancing tissue.
     - **Enhancing Tumor (ET):** Active contrast-enhancing rim.
  3. Precise volumes in cubic centimeters ($cm^3$) and anatomical multi-planar overlays (`Axial`, `Coronal`, `Sagittal`) are presented to the clinician.
  4. The clinician independently records a decision: **Accept**, **Modify**, **Reject**, or **Escalate**.

---

## 3. Strict Clinical Limitations & Safety Guardrails
1. **Not a Standalone Diagnostic Agent:** The segmentation masks and volume metrics are AI findings, not diagnostic confirmations, staging assessments, or therapeutic prescriptions.
2. **Mandatory Sequence Completeness:** The model strictly requires all 4 sequences (`T1`, `T1ce`, `T2`, `FLAIR`). If any sequence is absent or corrupted, the integration adapter enforces controlled rejection with `status: "insufficient_input"`. It is forbidden to synthesize or hallucinate missing sequences.
3. **Tumor Volume Interpretation:** Changes in tumor volume between scans cannot be autonomously interpreted as treatment response, recurrence, or pseudoprogression without clinician longitudinal context.
4. **Target Population:** Optimized for adult diffuse high-grade and low-grade gliomas with pre-operative imaging characteristics corresponding to the BraTS protocol.

---

## 4. Quantitative Performance Benchmarks (Held-Out Test Split)
Evaluated on the frozen, patient-level held-out test split of 16 patients from the authentic **BraTS-TCGA-GBM** cohort on NVIDIA GeForce RTX 4070 Laptop GPU (8GB VRAM):

| Subregion | Metric | Measured Value | PRD Success Threshold | Status |
|---|---|---|---|---|
| **Whole Tumor (WT)** | Mean Dice | **0.9171** (± 0.0481) | $\ge 0.80$ | **EXCEEDED (+11.7%)** |
| **Tumor Core (TC)** | Mean Dice | **0.9103** (± 0.0441) | $\ge 0.75$ | **EXCEEDED (+16.0%)** |
| **Enhancing Tumor (ET)** | Mean Dice | **0.8625** (± 0.0539) | $\ge 0.70$ | **EXCEEDED (+16.3%)** |
| **Inference Efficiency** | Latency / Volume | **2.28 s** | $< 3.0$ s | **PASSED** |

All metrics are verified and persisted in `glioma_ai/metrics.json`.
