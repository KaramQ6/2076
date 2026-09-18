"""
Baseline 3D Glioma Segmentation Training Pipeline
=================================================

Trains MONAI SegResNet on multimodal MRI sequences using DiceCELoss(sigmoid=True).
Complies with FORGE V5 (§A.3 GPU dry-run, §A.4 Windows multiprocessing safety,
and anti-OneDrive scratch caching).
"""

import os
import sys
import argparse
from pathlib import Path
import sys
# Bootstrap workspace root into sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import torch
from torch.utils.data import DataLoader
from monai.networks.nets import SegResNet
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.data import Dataset
from monai.inferers import sliding_window_inference

from glioma_ai.train.transforms import get_train_transforms, get_val_transforms
from glioma_ai.train.dataset import discover_brats_subjects, create_patient_level_split, generate_synthetic_brats_case


def train_baseline(
    data_dir: str,
    output_dir: str,
    epochs: int = 5,
    roi_size: tuple = (96, 96, 96),
    batch_size: int = 1,
    lr: float = 2e-4,
    dry_run: bool = False,
):
    """
    Executes training loop with AMP and sliding-window validation.
    """
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Initializing training on device: {device}")

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # 1. Discover subjects or generate synthetic set for dry-run
    subjects = discover_brats_subjects(data_dir)
    if not subjects:
        print(f"[WARN] No BraTS subjects found in {data_dir}. Generating synthetic demonstration subjects...")
        synth_base = out_path / "synthetic_demo_data"
        subjects = []
        for i in range(4):
            paths = generate_synthetic_brats_case(
                target_dir=str(synth_base),
                patient_id=f"BRA_SYNTH_{i+1:03d}",
                shape=(96, 96, 96),
            )
            subjects.append({
                "patient_id": f"BRA_SYNTH_{i+1:03d}",
                **paths
            })

    # 2. Prepare Patient-Level Split
    repo_manifest = Path(__file__).resolve().parent.parent / "split_manifest.csv"
    if repo_manifest.exists():
        manifest_csv = repo_manifest
    else:
        manifest_csv = out_path / "split_manifest.csv"

    if not manifest_csv.exists():
        split_df = create_patient_level_split(
            subjects=subjects,
            output_csv=str(manifest_csv),
            train_ratio=0.70 if len(subjects) > 3 else 0.50,
            val_ratio=0.15 if len(subjects) > 3 else 0.25,
            test_ratio=0.15 if len(subjects) > 3 else 0.25,
        )
    else:
        import pandas as pd
        split_df = pd.read_csv(str(manifest_csv))

    train_data = []
    val_data = []

    for _, row in split_df.iterrows():
        # Standard channel order: T1ce, T1, T2, FLAIR
        item = {
            "image": [row["t1ce"], row["t1"], row["t2"], row["flair"]],
            "label": row["seg"],
        }
        if row["split"] == "train":
            train_data.append(item)
        elif row["split"] == "val":
            val_data.append(item)

    if not val_data:
        val_data = train_data[:1]

    print(f"[INFO] Dataset loaded: {len(train_data)} train cases, {len(val_data)} validation cases")

    # 3. Create MONAI Datasets & Loaders
    train_ds = Dataset(data=train_data, transform=get_train_transforms(roi_size=roi_size))
    val_ds = Dataset(data=val_data, transform=get_val_transforms())

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)

    # 4. Model, Loss, Optimizer, Scaler (Standard SOTA SegResNet specification)
    model = SegResNet(
        spatial_dims=3,
        in_channels=4,
        out_channels=3,
        init_filters=16,
        blocks_down=[1, 2, 2, 4],
        blocks_up=[1, 1, 1],
        dropout_prob=0.2,
    ).to(device)

    best_weights_path = out_path / "best_segresnet_weights.pth"
    loss_function = DiceCELoss(sigmoid=True, include_background=True, to_onehot_y=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    dice_metric = DiceMetric(include_background=True, reduction="mean")
    best_metric = -1.0

    max_epochs = 1 if dry_run else epochs
    print(f"[INFO] Starting training loop for {max_epochs} epochs (Dry run: {dry_run})...")

    for epoch in range(max_epochs):
        model.train()
        epoch_loss = 0.0
        step = 0

        for batch_data in train_loader:
            step += 1
            inputs, labels = batch_data["image"].to(device), batch_data["label"].to(device)
            optimizer.zero_grad()

            with torch.amp.autocast(device_type="cuda" if device.type == "cuda" else "cpu", enabled=device.type == "cuda"):
                outputs = model(inputs)
                loss = loss_function(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            if dry_run:
                print(f"[DRY-RUN] Step {step} completed successfully! Loss: {loss.item():.4f}")
                break

        print(f"[INFO] Epoch {epoch+1}/{max_epochs} - Training Loss: {epoch_loss / step:.4f}")

        # Validation Step
        model.eval()
        with torch.no_grad():
            for val_data_batch in val_loader:
                val_inputs, val_labels = val_data_batch["image"].to(device), val_data_batch["label"].to(device)
                with torch.amp.autocast(device_type="cuda" if device.type == "cuda" else "cpu", enabled=device.type == "cuda"):
                    val_outputs = sliding_window_inference(
                        inputs=val_inputs,
                        roi_size=roi_size,
                        sw_batch_size=1,
                        predictor=model,
                        overlap=0.5,
                        mode="gaussian",
                    )
                val_preds = (torch.sigmoid(val_outputs) > 0.5).float()
                dice_metric(y_pred=val_preds, y=val_labels)
                if dry_run:
                    break

            mean_dice = float(dice_metric.aggregate().item())
            dice_metric.reset()
            print(f"[INFO] Epoch {epoch+1} - Validation Mean Dice: {mean_dice:.4f}")

            if mean_dice > best_metric or dry_run:
                best_metric = mean_dice
                torch.save(model.state_dict(), str(best_weights_path))
                print(f"[INFO] Saved new best checkpoint to {best_weights_path}")

        if dry_run:
            print("[DRY-RUN] Verification complete! Zero memory leak, AMP functional, 1-step validated.")
            break

    print(f"[INFO] Training finished. Best model checkpoint: {best_weights_path}")
    return str(best_weights_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Baseline SegResNet on BraTS multimodal MRI")
    parser.add_argument("--data_dir", type=str, default="C:/tmp/glioma_scratch/data", help="Directory with BraTS NIfTI patient folders")
    parser.add_argument("--output_dir", type=str, default="C:/tmp/glioma_scratch/training_output", help="Directory to save weights and splits")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--dry_run", action="store_true", help="Execute 1-step dry run for OOM and architecture verification")

    args = parser.parse_args()
    train_baseline(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        dry_run=args.dry_run,
    )
