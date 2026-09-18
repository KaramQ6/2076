# Data Card: BraTS-TCGA-GBM Baseline Dataset

## 1. Dataset Overview
- **Name:** BraTS-TCGA-GBM (Pre-operative Multimodal Brain MRI & Expert Segmentations)
- **Host:** The Cancer Imaging Archive (TCIA)
- **DOI:** [10.7937/K9/TCIA.2017.KLXWJJ1Q](https://doi.org/10.7937/K9/TCIA.2017.KLXWJJ1Q)
- **License:** Creative Commons Attribution (CC BY 3.0 / CC BY 4.0)
- **Cohort Size:** 102 confirmed pre-operative Glioblastoma (GBM) subjects
- **File Format:** Compressed NIfTI (`.nii.gz`)
- **Package Size:** ~767 MB

---

## 2. Imaging Sequences & Spatial Standards
Every subject comprises 4 rigid co-registered, skull-stripped, resampled MRI sequences:

| Sequence | Naming Convention | Description | Clinical Utility |
|---|---|---|---|
| **T1** | `*_t1.nii.gz` | Pre-contrast T1-weighted | Baseline anatomical architecture |
| **T1ce** | `*_t1ce.nii.gz` / `*_t1c.nii.gz` | Gadolinium-enhanced T1-weighted | Active blood-brain barrier breakdown (Enhancing Tumor) |
| **T2** | `*_t2.nii.gz` | T2-weighted | Peritumoral fluid and edema localization |
| **FLAIR** | `*_flair.nii.gz` | T2 Fluid Attenuated Inversion Recovery | Suppression of CSF signal; distinguishes edema from ventricles |

- **Volume Geometry:** 240 × 240 × 155 voxels
- **Voxel Spacing:** Isotropic 1.0 mm × 1.0 mm × 1.0 mm
- **Orientation:** Right-Anterior-Superior (RAS) anatomical coordinates

---

## 3. Ground Truth Annotation & Label Harmonization

### Raw Annotations
Ground truth segmentations (`*_seg.nii.gz`) were produced by computer-aided segmentation tools and manually corrected/approved by board-certified neuroradiologists:
- **0:** Healthy brain tissue / background
- **1:** Necrotic and non-enhancing tumor core (NCR/NET)
- **2:** Peritumoral edema (ED)
- **4:** GD-enhancing tumor (ET)
*(Note: Label 3 was historically deprecated in the BraTS TCGA-GBM cohort).*

### Evaluated Composite Regions
The segmentation task models the clinical composite targets:
1. **Tumor Core (TC):** Labels `1 + 4` (Necrotic core + Enhancing tumor)
2. **Whole Tumor (WT):** Labels `1 + 2 + 4` (All active tumor compartments and edema)
3. **Enhancing Tumor (ET):** Label `4` only (Hyperactive malignant border)

---

## 4. Leakage Prevention & Split Strategy
- **Partitioning Unit:** Pure **Patient-Level** (`patient_id`).
- **Forbidden Practice:** Any mixing of 2D axial slices or 3D patches from the same patient across splits is strictly prohibited.
- **Frozen Split:** Saved deterministically in `glioma_ai/split_manifest.csv` before model training.
- **Official Frozen Distribution (Seed 42):**
  * **Train:** 71 patients (69.6%)
  * **Validation:** 15 patients (14.7%)
  * **Held-Out Test Set:** 16 patients (15.7%)
- **Segmentation Provenance:** 97 subjects utilize neuroradiologist-verified manual corrections (`_GlistrBoost_ManuallyCorrected.nii.gz`), and 5 subjects utilize standard consensus segmentations (`_GlistrBoost.nii.gz`). All 102 cases strictly adhere to labels `[0, 1, 2, 4]`.
