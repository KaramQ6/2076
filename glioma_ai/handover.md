# Engineering Handover & Integration Guide
### National Oncology Intelligence Platform — Glioma AI Module
**Team Boom | Track: Irbid 2076 — Innovation in Healthcare Systems**

---

## 1. Executive Summary (The 3-Minute Team Briefing)
- **Clinical Mission:** Automated 3D segmentation and quantitative volumetry of glioma subregions to serve as an objective **AI Second Reader** for neuroradiologists and multidisciplinary tumor boards.
- **Data Reality:** Built upon TCIA `BraTS-TCGA-GBM` (102 pre-operative subjects, isotropic 1.0mm³ voxel resolution, rigid skull-stripped NIfTI).
- **Core Architecture:** `MONAI SegResNet 3D` with residual blocks, optimized via `DiceCELoss(sigmoid=True)` for overlapping multi-label regions.
- **Integration Principle:** Zero internal leakage to platform frontend; communicates strictly via the **Shared Result Envelope** with file-based artifact references.
- **Fail-Safe Contract:** Missing sequences immediately trigger `insufficient_input` with explanatory codes. Hallucination or caching stale successes is mathematically prohibited.

---

## 2. Package Architecture
```
glioma_ai/
├── README.md               # Quickstart guide
├── data_card.md            # Data provenance, label mapping, split manifest docs
├── split_manifest.csv      # Frozen patient-level train/val/test splits (zero slice leakage)
├── model_card.md           # Model architecture, hyperparams, clinical limitations
├── metrics.json            # Final held-out evaluation metrics (Dice, HD95, latency)
├── config.yaml             # Central configuration
├── handover.md             # This handover manual
├── weights/
│   └── best_segresnet_weights.pth  # Frozen model weights
├── train/
│   ├── dataset.py          # Data discovery, patient splitting, synthetic case generator
│   ├── transforms.py       # MONAI multi-channel pipeline with label standardization
│   └── train_baseline.py   # Training script with AMP and Windows multiprocessing safety
├── evaluation/
│   ├── evaluator.py        # Held-out split evaluation and metrics JSON generator
│   └── metrics.py          # Dice and HD95 multi-channel metrics
├── inference/
│   ├── predictor.py        # Sliding-window inference and NIfTI mask generator
│   └── overlay_generator.py# Axial, Coronal, Sagittal PNG overlays + cm³ volumetrics
├── adapter/
│   ├── glioma_adapter.py   # Main platform integration adapter class
│   └── schema.py           # Pydantic request and SharedResultEnvelope schemas
├── tests/
│   ├── test_transforms.py  # Unit tests for label remap and channel order
│   ├── test_schema.py      # Unit tests for Pydantic models
│   └── test_adapter.py     # Integration tests (success, missing sequence, bad geometry)
└── examples/
    ├── sample_result_envelope.json
    ├── input_manifest/
    ├── segmentation/
    └── overlays/
```

---

## 3. How to Run the Module

### A. Environment Prerequisites
```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install monai nibabel pydantic pyyaml pandas scikit-learn matplotlib
```

### B. Run Automated Verification Tests
```powershell
cmd /c "python -m pytest glioma_ai\tests -v"
```
*Expected: 100% tests passing across transforms, schemas, and adapter contracts.*

### C. Python Integration Call (How Platform Backend Calls the Module)
```python
from glioma_ai.adapter.glioma_adapter import GliomaPlatformAdapter

# 1. Initialize Adapter with frozen model weights
adapter = GliomaPlatformAdapter(
    weights_path="glioma_ai/weights/best_segresnet_weights.pth",
    artifacts_base_dir="C:/tmp/glioma_scratch/artifacts"
)

# 2. Prepare payload conforming to schema
payload = {
    "case_id": "GLIOMA-PATIENT-102",
    "cancer_type": "glioma",
    "modality": "MRI",
    "task": "glioma_subregion_segmentation",
    "sequences": {
        "t1": "path/to/patient_t1.nii.gz",
        "t1ce": "path/to/patient_t1ce.nii.gz",
        "t2": "path/to/patient_t2.nii.gz",
        "flair": "path/to/patient_flair.nii.gz"
    },
    "metadata": {
        "input_ref": "hakeem-case-2076-001"
    }
}

# 3. Execute inference
result_envelope = adapter.run_inference(payload)

# 4. Check status
if result_envelope["status"] == "succeeded":
    finding = result_envelope["findings"][0]
    print("Tumor Volumetrics:", finding["measurements"])
    print("Mask & Overlays:", finding["artifact_refs"])
elif result_envelope["status"] == "insufficient_input":
    print("Controlled Rejection:", result_envelope["error_code"], result_envelope["limitations"])
```

---

## 4. Contract Specifications: Input & Output Channels

### Input Sequences (Rigid 4-Channel Stacking)
1. `T1` (pre-contrast anatomical reference)
2. `T1ce` (post-contrast gadolinium active tumor enhancement)
3. `T2` (peritumoral fluid and edema)
4. `FLAIR` (ventricle suppression, edema demarcation)

### Output Channels (Multi-Label Sigmoid Activation)
- **Channel 0 — Tumor Core (TC):** Necrotic core + Enhancing tumor (`labels 1 + 4`)
- **Channel 1 — Whole Tumor (WT):** Necrotic core + Edema + Enhancing tumor (`labels 1 + 2 + 4`)
- **Channel 2 — Enhancing Tumor (ET):** Active malignant rim (`label 4`)

---

## 5. Failure Modes & Controlled Behavior

| Scenario | Input Condition | Adapter Response | Error Code |
|---|---|---|---|
| **Missing Sequence** | e.g., FLAIR absent or null | `status: "insufficient_input"` | `MISSING_REQUIRED_SEQUENCES` |
| **Geometry Mismatch** | Dimensions do not align (e.g. 240x240x155 vs 256x256x120) | `status: "failed"` | `INCOMPATIBLE_SEQUENCE_GEOMETRY` |
| **Corrupt File** | Invalid NIfTI header or read error | `status: "failed"` | `CORRUPT_NIFTI_FILE` |
| **Payload Schema Error** | Invalid JSON / missing fields | `status: "failed"` | `INVALID_PAYLOAD_SCHEMA` |

---

## 6. Checkpoint Verification Sign-Off
- [x] **Checkpoint 1 (Data & Task):** NIfTI discovery, 4-sequence verification, channel mapping frozen, `split_manifest.csv` locked at patient level.
- [x] **Checkpoint 2 (Real Inference & Metrics):** SegResNet 3D pipeline executable from clean command, AMP acceleration on CUDA, `metrics.json` recorded.
- [x] **Checkpoint 3 (Shared Platform Adapter):** Single stable entry point, 3-state envelope response (`succeeded` / `insufficient_input` / `failed`), clinical overlays (`axial`, `coronal`, `sagittal`) and cm³ measurements generated and verified.
