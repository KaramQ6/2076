"""
Dataset Loader, Splitter & Diagnostic Synthetic Data Generator
==============================================================

Provides strict patient-level dataset partitioning (zero leakage across slices),
file discovery for BraTS-TCGA-GBM NIfTI cases, and a synthetic 3D case generator
for deterministic pipeline verification and testing.
"""

import os
import glob
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
import nibabel as nib
from sklearn.model_selection import train_test_split


def discover_brats_subjects(data_dir: str) -> List[Dict[str, Any]]:
    """
    Scans data_dir for BraTS-TCGA-GBM patient directories and locates NIfTI sequences.
    
    Expected filenames per subject:
      * _t1.nii.gz
      * _t1ce.nii.gz (or _t1c.nii.gz)
      * _t2.nii.gz
      * _flair.nii.gz
      * _seg.nii.gz (optional during pure inference)
    """
    subjects = []
    base_path = Path(data_dir)
    if not base_path.exists():
        return subjects

    # If base_path has a single nested directory containing the patients
    subdirs = [d for d in base_path.iterdir() if d.is_dir()]
    if len(subdirs) == 1 and not any(subdirs[0].glob("*.nii*")):
        base_path = subdirs[0]

    # Look for patient folders
    for patient_dir in sorted(base_path.iterdir()):
        if not patient_dir.is_dir():
            continue
        
        patient_id = patient_dir.name
        nii_files = list(patient_dir.glob("*.nii*"))
        if not nii_files:
            continue

        subj_entry = {
            "patient_id": patient_id,
            "t1": None,
            "t1ce": None,
            "t2": None,
            "flair": None,
            "seg": None,
        }

        seg_candidates = []
        for f in nii_files:
            fn_lower = f.name.lower()
            if "t1ce" in fn_lower or "t1gd" in fn_lower or "t1c." in fn_lower:
                subj_entry["t1ce"] = str(f)
            elif "t1" in fn_lower and "t1ce" not in fn_lower and "t1gd" not in fn_lower:
                subj_entry["t1"] = str(f)
            elif "t2" in fn_lower and "flair" not in fn_lower:
                subj_entry["t2"] = str(f)
            elif "flair" in fn_lower:
                subj_entry["flair"] = str(f)
            elif "manuallycorrected" in fn_lower:
                seg_candidates.insert(0, str(f)) # Highest priority: expert manual correction
            elif "glistrboost" in fn_lower or "seg" in fn_lower or "label" in fn_lower:
                seg_candidates.append(str(f))

        if seg_candidates:
            subj_entry["seg"] = seg_candidates[0]

        subjects.append(subj_entry)

    return subjects


def create_patient_level_split(
    subjects: List[Dict[str, Any]],
    output_csv: str,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Creates a strict patient-level split (zero data leakage) and freezes it to CSV.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-5, "Ratios must sum to 1.0"
    
    patient_ids = [s["patient_id"] for s in subjects]
    
    # First split: Train vs Temp (Val + Test)
    temp_ratio = val_ratio + test_ratio
    train_ids, temp_ids = train_test_split(
        patient_ids, test_size=temp_ratio, random_state=seed, shuffle=True
    )
    
    # Second split: Val vs Test
    val_rel_ratio = val_ratio / temp_ratio
    val_ids, test_ids = train_test_split(
        temp_ids, test_size=(1.0 - val_rel_ratio), random_state=seed, shuffle=True
    )

    records = []
    for s in subjects:
        pid = s["patient_id"]
        split = "train" if pid in train_ids else ("val" if pid in val_ids else "test")
        records.append({
            "patient_id": pid,
            "split": split,
            "t1": s.get("t1"),
            "t1ce": s.get("t1ce"),
            "t2": s.get("t2"),
            "flair": s.get("flair"),
            "seg": s.get("seg"),
        })

    df = pd.DataFrame(records)
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    return df


def generate_synthetic_brats_case(
    target_dir: str,
    patient_id: str = "SYNTH_PATIENT_01",
    shape: Tuple[int, int, int] = (128, 128, 128),
    spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> Dict[str, str]:
    """
    Generates an anatomically structured synthetic 3D multimodal brain volume
    with an embedded multi-compartment glioma lesion:
      - Label 1: Necrotic Core (center)
      - Label 2: Edema (outer boundary)
      - Label 4: Enhancing Tumor (ring surrounding necrotic core)
    
    Used for dry-runs, GPU sanity checks, and platform integration tests.
    """
    out_dir = Path(target_dir) / patient_id
    out_dir.mkdir(parents=True, exist_ok=True)

    d, h, w = shape
    affine = np.diag([spacing[0], spacing[1], spacing[2], 1.0])

    # 1. Generate brain background (ellipsoid)
    z, y, x = np.ogrid[:d, :h, :w]
    cz, cy, cx = d // 2, h // 2, w // 2
    brain_mask = ((z - cz) ** 2 / (d * 0.4) ** 2 +
                  (y - cy) ** 2 / (h * 0.4) ** 2 +
                  (x - cx) ** 2 / (w * 0.35) ** 2) <= 1.0

    # 2. Generate tumor subregions in left hemisphere
    tz, ty, tx = cz, cy + h // 6, cx + w // 6
    r_edema = min(d, h, w) * 0.18
    r_core = r_edema * 0.55
    r_necrotic = r_core * 0.35

    dist_tumor = np.sqrt((z - tz) ** 2 + (y - ty) ** 2 + (x - tx) ** 2)

    seg = np.zeros(shape, dtype=np.int16)
    # Edema = 2
    seg[dist_tumor <= r_edema] = 2
    # Enhancing Tumor = 4
    seg[dist_tumor <= r_core] = 4
    # Necrotic Core = 1
    seg[dist_tumor <= r_necrotic] = 1
    # Restrict to brain
    seg[~brain_mask] = 0

    # 3. Create simulated multi-sequence image intensities
    # Base brain tissue intensities
    base_brain = np.zeros(shape, dtype=np.float32)
    base_brain[brain_mask] = 50.0 + np.random.normal(0, 5, size=np.sum(brain_mask))

    # T1: baseline tissue contrast, tumor is hypointense
    img_t1 = base_brain.copy()
    img_t1[seg == 1] *= 0.6
    img_t1[seg == 2] *= 0.8
    img_t1[seg == 4] *= 0.9

    # T1ce: enhancing tumor (label 4) lights up brightly!
    img_t1ce = img_t1.copy()
    img_t1ce[seg == 4] += 80.0

    # T2: edema (label 2) is hyperintense (bright)
    img_t2 = base_brain.copy()
    img_t2[seg == 2] += 90.0
    img_t2[seg == 1] += 70.0
    img_t2[seg == 4] += 60.0

    # FLAIR: edema and non-enhancing core are bright, cerebrospinal fluid suppressed
    img_flair = base_brain.copy()
    img_flair[seg == 2] += 110.0
    img_flair[seg == 4] += 70.0

    # Add realistic background noise
    for img in [img_t1, img_t1ce, img_t2, img_flair]:
        img += np.random.normal(0, 2, size=shape).astype(np.float32)
        img[img < 0] = 0

    # Save to NIfTI
    paths = {}
    seq_map = {
        "t1": (img_t1, f"{patient_id}_t1.nii.gz"),
        "t1ce": (img_t1ce, f"{patient_id}_t1ce.nii.gz"),
        "t2": (img_t2, f"{patient_id}_t2.nii.gz"),
        "flair": (img_flair, f"{patient_id}_flair.nii.gz"),
        "seg": (seg, f"{patient_id}_seg.nii.gz"),
    }

    for key, (data, filename) in seq_map.items():
        fp = out_dir / filename
        nii = nib.Nifti1Image(data, affine)
        nib.save(nii, str(fp))
        paths[key] = str(fp)

    return paths
