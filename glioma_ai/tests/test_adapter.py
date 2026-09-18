"""
Integration tests for GliomaPlatformAdapter
===========================================
Validates:
  1. Controlled rejection on missing sequence (status="insufficient_input").
  2. Controlled rejection on geometry mismatch (status="failed").
  3. Full successful inference run returning valid SharedResultEnvelope,
     tumor volume measurements (cm³), and clinical overlays.
"""

import os
import shutil
import pytest
from pathlib import Path
import nibabel as nib
import numpy as np

from glioma_ai.adapter.glioma_adapter import GliomaPlatformAdapter
from glioma_ai.train.dataset import generate_synthetic_brats_case


@pytest.fixture(scope="module")
def synthetic_case_dir():
    """Generates a temporary synthetic 3D multi-sequence patient case."""
    test_dir = Path("C:/tmp/glioma_scratch/test_fixtures")
    test_dir.mkdir(parents=True, exist_ok=True)
    paths = generate_synthetic_brats_case(
        target_dir=str(test_dir),
        patient_id="TEST_PT_001",
        shape=(96, 96, 96), # Compact size for rapid unit test
    )
    yield paths
    # Teardown
    shutil.rmtree(str(test_dir / "TEST_PT_001"), ignore_errors=True)


def test_adapter_missing_sequence_insufficient_input(synthetic_case_dir):
    """Verify that omitting a sequence gracefully returns status='insufficient_input'."""
    adapter = GliomaPlatformAdapter(artifacts_base_dir="C:/tmp/glioma_scratch/test_artifacts")
    
    # Intentionally omit FLAIR
    payload = {
        "case_id": "TEST_PT_001",
        "sequences": {
            "t1": synthetic_case_dir["t1"],
            "t1ce": synthetic_case_dir["t1ce"],
            "t2": synthetic_case_dir["t2"],
            "flair": None, # Missing!
        },
        "metadata": {"input_ref": "test_missing_flair"},
    }

    result = adapter.run_inference(payload)
    assert result["status"] == "insufficient_input"
    assert result["quality_status"] == "failed"
    assert result["error_code"] == "MISSING_REQUIRED_SEQUENCES"
    assert len(result["findings"]) == 0
    assert "Missing required MRI sequence(s): flair" in result["limitations"][0]


def test_adapter_incompatible_geometry_failure(synthetic_case_dir):
    """Verify that passing mismatched spatial dimensions gracefully returns status='failed'."""
    test_dir = Path("C:/tmp/glioma_scratch/test_fixtures/mismatched_pt")
    test_dir.mkdir(parents=True, exist_ok=True)
    
    # Create an incompatible sequence with shape (64, 64, 64)
    mismatched_file = test_dir / "bad_flair.nii.gz"
    nib.save(
        nib.Nifti1Image(np.zeros((64, 64, 64), dtype=np.float32), np.eye(4)),
        str(mismatched_file),
    )

    adapter = GliomaPlatformAdapter(artifacts_base_dir="C:/tmp/glioma_scratch/test_artifacts")
    payload = {
        "case_id": "TEST_PT_002",
        "sequences": {
            "t1": synthetic_case_dir["t1"],
            "t1ce": synthetic_case_dir["t1ce"],
            "t2": synthetic_case_dir["t2"],
            "flair": str(mismatched_file),
        },
        "metadata": {"input_ref": "test_mismatched_geometry"},
    }

    result = adapter.run_inference(payload)
    assert result["status"] == "failed"
    assert result["error_code"] == "INCOMPATIBLE_SEQUENCE_GEOMETRY"

    shutil.rmtree(str(test_dir), ignore_errors=True)


def test_adapter_full_inference_success(synthetic_case_dir):
    """Verify full end-to-end execution producing measurements, mask, and overlays."""
    adapter = GliomaPlatformAdapter(artifacts_base_dir="C:/tmp/glioma_scratch/test_artifacts")
    payload = {
        "case_id": "TEST_PT_001",
        "sequences": {
            "t1": synthetic_case_dir["t1"],
            "t1ce": synthetic_case_dir["t1ce"],
            "t2": synthetic_case_dir["t2"],
            "flair": synthetic_case_dir["flair"],
        },
        "metadata": {"input_ref": "test_full_success"},
    }

    result = adapter.run_inference(payload)

    assert result["status"] == "succeeded"
    assert result["quality_status"] == "passed"
    assert result["error_code"] is None
    assert len(result["findings"]) == 1

    finding = result["findings"][0]
    assert finding["label"] == "glioma_subregion_segmentation"
    
    # Check measurements
    meas_names = [m["name"] for m in finding["measurements"]]
    assert "whole_tumor_volume" in meas_names
    assert "tumor_core_volume" in meas_names
    assert "enhancing_tumor_volume" in meas_names

    # Check generated artifacts
    artifacts = finding["artifact_refs"]
    assert len(artifacts) >= 4 # NIfTI mask + 3 overlay PNGs
    for art in artifacts:
        assert os.path.exists(art), f"Artifact file does not exist: {art}"
