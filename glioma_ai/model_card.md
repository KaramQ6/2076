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

---

## 5. Qualitative Evidence & Representative Clinical Cases

In compliance with Page 6 of the Glioma AI Module Brief, three representative prediction cases have been evaluated and persisted with full 3-plane anatomical overlays in `glioma_ai/examples/overlays/` and volumetric segmentation masks in `glioma_ai/examples/segmentation/`:

1. **Strong Exemplar (`TCGA-02-0033`):**
   - **Metrics:** WT Dice: **0.9425** | TC Dice: **0.9542** | ET Dice: **0.9185** | Latency: **1.79s**.
   - **Tumor Volumes:** WT: $101.34\text{ cm}^3$, TC: $41.69\text{ cm}^3$, ET: $32.40\text{ cm}^3$.
   - **Clinical Profile:** Classical glioblastoma with distinct thick nodular peripheral ring enhancement around a necrotic center. Clear delineation from brain parenchyma.
   - **Artifacts:** `glioma_ai/examples/overlays/TCGA-02-0033_overlay_{axial,coronal,sagittal}.png`.

2. **Difficult / Heterogeneous Exemplar (`TCGA-06-0149`):**
   - **Metrics:** WT Dice: **0.7805** | TC Dice: **0.8232** | ET Dice: **0.7647** | Latency: **1.72s**.
   - **Tumor Volumes:** WT: $61.74\text{ cm}^3$, TC: $39.46\text{ cm}^3$, ET: $15.82\text{ cm}^3$.
   - **Clinical Profile:** Infiltrative, multi-focal diffuse edema with non-uniform signal on T2/FLAIR and subtle non-continuous enhancement on T1ce.
   - **Artifacts:** `glioma_ai/examples/overlays/TCGA-06-0149_overlay_{axial,coronal,sagittal}.png`.

3. **Under-Segmentation / Failure Pattern Exemplar (`TCGA-19-5954`):**
   - **Metrics:** WT Dice: **0.9140** | TC Dice: **0.8101** | ET Dice: **0.7219** | Latency: **2.81s**.
   - **Tumor Volumes:** WT: $47.53\text{ cm}^3$, TC: $23.11\text{ cm}^3$, ET: $12.04\text{ cm}^3$.
   - **Clinical Profile:** Paucicellular necrotic region with very fine, scattered micro-enhancing foci. The model successfully captures the macroscopic edema/WT boundary, but under-segments the finest discontinuous micro-enhancing islands on ET due to voxel resolution thresholds.
   - **Safety Guardrail:** Clinician review workflow surfaces both the overlay and the volume breakdown, enabling the neuroradiologist to manually expand the ET boundary before sign-off.
   - **Artifacts:** `glioma_ai/examples/overlays/TCGA-19-5954_overlay_{axial,coronal,sagittal}.png`.

