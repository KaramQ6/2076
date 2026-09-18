# Changelog

All notable changes to this project will be documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Complete `glioma_ai` package structure and Pydantic schemas for Shared Result Envelope.
- `StandardizeBraTSLabelsd` transform and MONAI multi-channel BraTS pipeline.
- `GliomaPlatformAdapter` with fail-safe contract for missing sequences and geometric mismatches.
- Sliding window inference engine with 3-plane (axial, coronal, sagittal) overlay generation and physical volume ($cm^3$) calculations.
- Comprehensive unit and integration test suite with 100% pass rate.
- Initial baseline training and dry-run pipeline with AMP acceleration.
