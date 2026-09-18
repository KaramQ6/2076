"""
Unit tests for platform integration schemas
===========================================
Validates Pydantic request models, SharedResultEnvelope serialization, and fail-safe behavior.
"""

import json
import pytest
from glioma_ai.adapter.schema import (
    GliomaInferenceRequest,
    SequencePaths,
    SharedResultEnvelope,
    Finding,
    Measurement,
)


def test_valid_request_serialization():
    """Verify standard valid request parsing."""
    req_data = {
        "case_id": "TCGA-02-0001",
        "sequences": {
            "t1": "/data/tcga_01_t1.nii.gz",
            "t1ce": "/data/tcga_01_t1ce.nii.gz",
            "t2": "/data/tcga_01_t2.nii.gz",
            "flair": "/data/tcga_01_flair.nii.gz",
        },
    }
    req = GliomaInferenceRequest(**req_data)
    assert req.case_id == "TCGA-02-0001"
    assert req.cancer_type == "glioma"
    assert req.modality == "MRI"
    assert req.sequences.t1ce == "/data/tcga_01_t1ce.nii.gz"


def test_shared_result_envelope_succeeded():
    """Verify standard successful result envelope matches required fields."""
    envelope = SharedResultEnvelope(
        run_id="RUN-GLI-2026-001",
        case_id="TCGA-02-0001",
        input_ref="manifest-entry-01",
        model_id="glioma_segresnet_3d",
        model_version="1.0.0",
        status="succeeded",
        quality_status="passed",
        findings=[
            Finding(
                finding_id="FIND-01",
                measurements=[
                    Measurement(name="whole_tumor_volume", value=48.25, unit="cm3"),
                    Measurement(name="tumor_core_volume", value=22.10, unit="cm3"),
                    Measurement(name="enhancing_tumor_volume", value=12.40, unit="cm3"),
                ],
                artifact_refs=["segmentation_mask.nii.gz", "overlay_axial.png"],
            )
        ],
    )
    dumped = envelope.model_dump()
    assert dumped["status"] == "succeeded"
    assert len(dumped["findings"]) == 1
    assert dumped["findings"][0]["measurements"][0]["value"] == 48.25
    assert "Requires clinician review" in dumped["limitations"][0]


def test_shared_result_envelope_insufficient_input():
    """Verify controlled rejection envelope when sequences are missing."""
    envelope = SharedResultEnvelope(
        run_id="RUN-GLI-2026-002",
        case_id="TCGA-02-0002",
        input_ref="manifest-entry-02",
        model_id="glioma_segresnet_3d",
        model_version="1.0.0",
        status="insufficient_input",
        quality_status="failed",
        findings=[],
        error_code="MISSING_REQUIRED_SEQUENCE",
        limitations=["Missing required FLAIR sequence."],
    )
    assert envelope.status == "insufficient_input"
    assert envelope.error_code == "MISSING_REQUIRED_SEQUENCE"
    assert len(envelope.findings) == 0
