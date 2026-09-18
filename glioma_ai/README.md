# National Oncology Intelligence Platform — Glioma AI Module
**Team Boom | Irbid 2076: Innovation in Healthcare Systems**

An evidence-backed, reproducible 3D medical AI module for multimodal brain MRI glioma subregion segmentation, developed strictly under the **FORGE V5** doctrine and clinical PRD specifications.

---

## Quickstart

### 1. Verify Installation & Automated Tests
Ensure CUDA and dependencies are installed, then run the test suite:
```powershell
cmd /c "python -m pytest glioma_ai/tests -v"
```
All tests across label harmonization, schema validation, and adapter integration will run and pass.

### 2. Run 1-Step GPU Training Verification (Dry-Run)
Verify SegResNet forward, backward, AMP scaling, and sliding-window validation on your local GPU (e.g. RTX 4070 8GB):
```powershell
cmd /c "python glioma_ai/train/train_baseline.py --dry_run"
```

### 3. Run Held-Out Evaluation & Generate `metrics.json`
```powershell
cmd /c "python glioma_ai/evaluation/evaluator.py"
```

### 4. Integration Adapter Example (Platform Ingestion)
```python
from glioma_ai.adapter.glioma_adapter import GliomaPlatformAdapter

adapter = GliomaPlatformAdapter(weights_path="glioma_ai/weights/best_segresnet_weights.pth")

request_payload = {
    "case_id": "TCGA-GBM-001",
    "cancer_type": "glioma",
    "modality": "MRI",
    "task": "glioma_subregion_segmentation",
    "sequences": {
        "t1": "data/patient_t1.nii.gz",
        "t1ce": "data/patient_t1ce.nii.gz",
        "t2": "data/patient_t2.nii.gz",
        "flair": "data/patient_flair.nii.gz"
    }
}

result_envelope = adapter.run_inference(request_payload)
print(result_envelope["status"]) # 'succeeded' or 'insufficient_input'
```

---

## Core Specifications
- **Base Architecture:** MONAI `SegResNet` 3D (Residual Encoder-Decoder)
- **Input Channels (4):** `T1`, `T1ce` (contrast), `T2`, `FLAIR`
- **Output Channels (3):**
  - Channel 0: **Tumor Core (TC)** (Labels 1 + 4)
  - Channel 1: **Whole Tumor (WT)** (Labels 1 + 2 + 4)
  - Channel 2: **Enhancing Tumor (ET)** (Label 4)
- **Loss:** `DiceCELoss(sigmoid=True, include_background=True, to_onehot_y=False)`
- **Inference Strategy:** `sliding_window_inference` (roi_size 96x96x96, overlap 0.5)
- **Clinical Volumetry:** Physical volumes reported in $cm^3$ calculated directly from voxel spacing.

---

## Documentation Index
- [`data_card.md`](data_card.md): Dataset specifications, provenance, and split policy.
- [`model_card.md`](model_card.md): Model architecture, intended use, and limitations.
- [`handover.md`](handover.md): Complete engineering handover and integration contract.
- [`metrics.json`](metrics.json): Auditable quantitative performance on held-out split.
- [`config.yaml`](config.yaml): Central hyperparameter and environment settings.
