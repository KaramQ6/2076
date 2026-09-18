"""
Pydantic Schemas for Platform Integration Contract
===================================================

Defines the rigid request and result envelope schemas connecting the Glioma AI
module with the shared National Oncology Intelligence Platform.
"""

from typing import List, Optional, Dict, Any, Literal
from datetime import datetime, timezone
from pydantic import BaseModel, Field


class SequencePaths(BaseModel):
    """File paths for the 4 required MRI sequences."""
    t1: Optional[str] = Field(None, description="Path to pre-contrast T1-weighted NIfTI volume")
    t1ce: Optional[str] = Field(None, description="Path to gadolinium-enhanced T1-weighted NIfTI volume (T1-Gd)")
    t2: Optional[str] = Field(None, description="Path to T2-weighted NIfTI volume")
    flair: Optional[str] = Field(None, description="Path to T2-FLAIR NIfTI volume")


class GliomaInferenceRequest(BaseModel):
    """Inference request payload submitted from the platform."""
    case_id: str = Field(..., description="Unique patient case identifier")
    cancer_type: Literal["glioma"] = "glioma"
    modality: Literal["MRI"] = "MRI"
    task: Literal["glioma_subregion_segmentation"] = "glioma_subregion_segmentation"
    sequences: SequencePaths = Field(..., description="Paths to required MRI sequences")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Supplementary clinical metadata")


class Measurement(BaseModel):
    """Quantitative anatomical tumor metric derived from segmentation mask."""
    name: str = Field(..., description="Measurement metric name, e.g., whole_tumor_volume")
    value: float = Field(..., description="Computed metric value")
    unit: str = Field("cm3", description="Physical measurement unit")


class Finding(BaseModel):
    """Structured AI segmentation finding pending clinician second-reader review."""
    finding_id: str = Field(..., description="Unique finding ID")
    label: str = Field("glioma_segmentation", description="Finding label descriptor")
    measurements: List[Measurement] = Field(default_factory=list, description="Computed volumetric measurements")
    score: Optional[float] = Field(None, description="Confidence/Dice evaluation proxy if applicable; null by default")
    artifact_refs: List[str] = Field(default_factory=list, description="Relative or absolute paths to NIfTI masks and visualization overlays")


class SharedResultEnvelope(BaseModel):
    """
    Official Platform Integration Result Envelope.
    
    Why: Keeps disease-specific AI models decoupled from the host platform while
    guaranteeing auditable, reproducible, and verifiable result exchange.
    """
    schema_version: str = Field("1.0", description="Contract schema version")
    run_id: str = Field(..., description="Unique execution run identifier")
    case_id: str = Field(..., description="Patient case ID")
    input_ref: str = Field(..., description="Reference to input manifest or session")
    cancer_type: str = Field("glioma", description="Disease domain")
    modality: str = Field("MRI", description="Imaging modality")
    task: str = Field("glioma_subregion_segmentation", description="Specific segmentation task")
    model_id: str = Field(..., description="Model identifier")
    model_version: str = Field(..., description="Model version tag")
    generated_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="ISO-8601 UTC timestamp of result generation",
    )
    execution_mode: Literal["live", "cached_fallback"] = Field("live", description="Execution mode")
    status: Literal["succeeded", "insufficient_input", "failed"] = Field(
        ..., description="Execution status"
    )
    quality_status: Optional[Literal["passed", "failed", "unverified"]] = Field(
        "passed", description="Quality verification check"
    )
    findings: List[Finding] = Field(default_factory=list, description="List of generated findings")
    error_code: Optional[str] = Field(None, description="Explicit failure reason code")
    limitations: List[str] = Field(
        default_factory=lambda: [
            "Requires clinician review; segmentation is an AI finding, not a clinical treatment decision.",
            "Valid only for co-registered skull-stripped multimodal MRI (T1, T1ce, T2, FLAIR).",
        ],
        description="Documented medical and operational limitations",
    )
