"""
Platform Integration Adapter for Glioma AI Module
=================================================

Provides a stable, audited entry point (`run_glioma_inference`) conforming to the
National Oncology Intelligence Platform Shared Result Envelope contract.
"""

import os
import uuid
from pathlib import Path
from typing import Dict, Any, Optional
import nibabel as nib
from pydantic import ValidationError

from glioma_ai.adapter.schema import (
    GliomaInferenceRequest,
    SharedResultEnvelope,
    Finding,
    Measurement,
)
from glioma_ai.inference.predictor import GliomaPredictor


class GliomaPlatformAdapter:
    """
    Adapter implementing the official integration contract.
    """

    MODEL_ID = "glioma_segresnet_3d"
    MODEL_VERSION = "1.0.0"

    def __init__(
        self,
        weights_path: Optional[str] = None,
        artifacts_base_dir: str = "C:\\tmp\\glioma_scratch\\artifacts",
    ):
        self.artifacts_base_dir = Path(artifacts_base_dir)
        self.artifacts_base_dir.mkdir(parents=True, exist_ok=True)
        self.predictor = GliomaPredictor(weights_path=weights_path)

    def run_inference(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main entry point for platform requests.
        
        Guarantees:
          - Always returns a valid SharedResultEnvelope dictionary.
          - Missing sequences strictly yield status="insufficient_input".
          - Mismatched geometries yield status="failed".
          - Successful runs yield status="succeeded" with reviewable artifacts.
        """
        run_id = f"RUN-GLI-{uuid.uuid4().hex[:8].upper()}"

        # 1. Schema Validation
        try:
            req = GliomaInferenceRequest(**payload)
        except ValidationError as e:
            envelope = SharedResultEnvelope(
                run_id=run_id,
                case_id=payload.get("case_id", "UNKNOWN"),
                input_ref=str(payload.get("metadata", {}).get("input_ref", "direct_call")),
                model_id=self.MODEL_ID,
                model_version=self.MODEL_VERSION,
                status="failed",
                quality_status="failed",
                error_code="INVALID_PAYLOAD_SCHEMA",
                findings=[],
                limitations=[f"Payload schema validation error: {str(e)}"],
            )
            return envelope.model_dump()

        case_id = req.case_id
        input_ref = str(req.metadata.get("input_ref", f"case-{case_id}"))

        # 2. Required Sequences Validation
        seqs = req.sequences
        missing = []
        for name in ["t1", "t1ce", "t2", "flair"]:
            val = getattr(seqs, name, None)
            if not val or not os.path.exists(val):
                missing.append(name)

        if missing:
            envelope = SharedResultEnvelope(
                run_id=run_id,
                case_id=case_id,
                input_ref=input_ref,
                model_id=self.MODEL_ID,
                model_version=self.MODEL_VERSION,
                status="insufficient_input",
                quality_status="failed",
                error_code="MISSING_REQUIRED_SEQUENCES",
                findings=[],
                limitations=[
                    f"Missing required MRI sequence(s): {', '.join(missing)}. "
                    "All 4 co-registered sequences (T1, T1ce, T2, FLAIR) must be present for accurate subregion segmentation."
                ],
            )
            return envelope.model_dump()

        # 3. Geometry and Dimension Compatibility Check
        seq_dict = {
            "t1": seqs.t1,
            "t1ce": seqs.t1ce,
            "t2": seqs.t2,
            "flair": seqs.flair,
        }

        shapes = {}
        for name, path in seq_dict.items():
            try:
                header = nib.load(path).header
                shapes[name] = header.get_data_shape()
            except Exception as ex:
                envelope = SharedResultEnvelope(
                    run_id=run_id,
                    case_id=case_id,
                    input_ref=input_ref,
                    model_id=self.MODEL_ID,
                    model_version=self.MODEL_VERSION,
                    status="failed",
                    quality_status="failed",
                    error_code="CORRUPT_NIFTI_FILE",
                    findings=[],
                    limitations=[f"Failed to read header for sequence {name}: {str(ex)}"],
                )
                return envelope.model_dump()

        # Ensure all shapes match
        ref_shape = shapes["t1"]
        mismatched = [k for k, s in shapes.items() if s != ref_shape]
        if mismatched:
            envelope = SharedResultEnvelope(
                run_id=run_id,
                case_id=case_id,
                input_ref=input_ref,
                model_id=self.MODEL_ID,
                model_version=self.MODEL_VERSION,
                status="failed",
                quality_status="failed",
                error_code="INCOMPATIBLE_SEQUENCE_GEOMETRY",
                findings=[],
                limitations=[
                    f"Sequence spatial dimensions do not align: reference T1={ref_shape}, "
                    f"mismatched sequences: { {k: shapes[k] for k in mismatched} }"
                ],
            )
            return envelope.model_dump()

        # 4. Run Model Prediction
        case_out_dir = self.artifacts_base_dir / case_id / run_id
        case_out_dir.mkdir(parents=True, exist_ok=True)

        try:
            pred_res = self.predictor.predict(
                sequence_paths=seq_dict,
                output_dir=str(case_out_dir),
                case_id=case_id,
            )
        except Exception as ex:
            envelope = SharedResultEnvelope(
                run_id=run_id,
                case_id=case_id,
                input_ref=input_ref,
                model_id=self.MODEL_ID,
                model_version=self.MODEL_VERSION,
                status="failed",
                quality_status="failed",
                error_code="INFERENCE_EXECUTION_FAILURE",
                findings=[],
                limitations=[f"Inference pipeline execution error: {str(ex)}"],
            )
            return envelope.model_dump()

        # 5. Build Succeeded Shared Result Envelope
        measurements = [
            Measurement(name=m["name"], value=m["value"], unit=m["unit"])
            for m in pred_res["measurements"]
        ]

        finding = Finding(
            finding_id=f"FIND-{case_id}-01",
            label="glioma_subregion_segmentation",
            measurements=measurements,
            score=None, # Clinical requirement: Dice is an evaluation metric, not diagnostic certainty
            artifact_refs=pred_res["artifact_refs"],
        )

        envelope = SharedResultEnvelope(
            run_id=run_id,
            case_id=case_id,
            input_ref=input_ref,
            model_id=self.MODEL_ID,
            model_version=self.MODEL_VERSION,
            status="succeeded",
            quality_status="passed",
            findings=[finding],
            error_code=None,
            limitations=[
                "Requires clinician second-reader review before any therapeutic decision.",
                f"Computed volume measurements derived from voxel spacing: {pred_res['voxel_spacing']}.",
            ],
        )

        return envelope.model_dump()
