# Project Guidelines — Glioma AI Module
**National Oncology Intelligence Platform (Team Boom - Irbid 2076)**

## Stack & Environment
- **Runtime:** Python 3.11, PyTorch 2.7.0+cu126, MONAI 1.6.0
- **Hardware Profile:** NVIDIA GeForce RTX 4070 Laptop GPU (8GB VRAM), CUDA 13.2
- **Key Libraries:** `monai`, `nibabel`, `pydantic>=2.0`, `pyyaml`, `scikit-learn`, `matplotlib`

## Standard Commands
- Run test suite: `python -m pytest glioma_ai/tests -v`
- Run GPU dry-run / training: `python glioma_ai/train/train_baseline.py --dry_run`
- Run evaluation on held-out test split: `python glioma_ai/evaluation/evaluator.py`

## Operating Rules & Constraints (FORGE V5 & Engineering OS)
1. **Anti-OneDrive Cache:** Never store intermediate NIfTI volumes, checkpoints, or active training runs inside the OneDrive-synced repo folder. Route all heavy I/O to `C:\tmp\glioma_scratch\`.
2. **Windows Multiprocessing:** Always use `num_workers=0` in DataLoaders and wrap scripts with `if __name__ == '__main__':` to avoid multiprocessing spawn hangs.
3. **Labels & Channels:**
   - Raw BraTS: 0=Background, 1=NCR/NET, 2=Edema, 4=ET.
   - Output Channels: `0: TC (1+4)`, `1: WT (1+2+4)`, `2: ET (4)`.
   - Loss Function: `DiceCELoss(sigmoid=True, include_background=True, to_onehot_y=False)`.
4. **Adapter Invariance:** `GliomaPlatformAdapter.run_inference()` must strictly return the canonical `SharedResultEnvelope`. Missing sequences must gracefully yield `status: "insufficient_input"` with zero hallucination.
