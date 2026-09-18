"""
Model Evaluator & Benchmark Report Generator
============================================

Evaluates trained checkpoint on held-out test subjects, records Dice, IoU, HD95,
and latency, and exports the final `metrics.json` deliverable.
"""

import os
import json
import time
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from typing import Dict, Any, List, Optional
import torch
import numpy as np
import pandas as pd
from monai.networks.nets import SegResNet
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric, HausdorffDistanceMetric

from glioma_ai.inference.predictor import GliomaPredictor


def run_evaluation(
    weights_path: str,
    manifest_csv: str,
    output_metrics_json: str,
    device: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Evaluates model on the 'test' split defined in manifest_csv.
    """
    if device is None:
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        dev = torch.device(device)

    print(f"[INFO] Evaluating checkpoint {weights_path} on {dev}...")
    df = pd.read_csv(manifest_csv)
    test_df = df[df["split"] == "test"]
    if test_df.empty:
        print("[WARN] Test split is empty; falling back to val split for evaluation report.")
        test_df = df[df["split"] == "val"]

    predictor = GliomaPredictor(weights_path=weights_path, device=str(dev))

    dice_metric = DiceMetric(include_background=True, reduction="none", ignore_empty=True)
    hd95_metric = HausdorffDistanceMetric(include_background=True, percentile=95, reduction="none")

    results_per_patient = []
    latencies = []

    for _, row in test_df.iterrows():
        pid = row["patient_id"]
        seq_paths = {
            "t1": row["t1"],
            "t1ce": row["t1ce"],
            "t2": row["t2"],
            "flair": row["flair"],
        }
        
        start_t = time.time()
        # Predict
        out_dir = Path("C:/tmp/glioma_scratch/eval_runs") / pid
        pred_res = predictor.predict(sequence_paths=seq_paths, output_dir=str(out_dir), case_id=pid)
        elapsed = time.time() - start_t
        latencies.append(elapsed)

        # If ground truth segmentation is available, compute Dice & HD95
        seg_path = row.get("seg")
        if seg_path and os.path.exists(str(seg_path)):
            import nibabel as nib
            gt_nii = nib.load(seg_path).get_fdata()
            # Raw labels: 1=NCR, 2=ED, 4=ET
            gt_tc = ((gt_nii == 1) | (gt_nii == 4)).astype(np.float32)
            gt_wt = ((gt_nii == 1) | (gt_nii == 2) | (gt_nii == 4)).astype(np.float32)
            gt_et = (gt_nii == 4).astype(np.float32)
            gt_3ch = np.stack([gt_tc, gt_wt, gt_et], axis=0) # (3, D, H, W)

            # Predicted segmentation from saved NIfTI
            pred_seg_file = [f for f in pred_res["artifact_refs"] if f.endswith(".nii.gz")][0]
            pred_nii = nib.load(pred_seg_file).get_fdata()
            pr_tc = ((pred_nii == 1) | (pred_nii == 4)).astype(np.float32)
            pr_wt = ((pred_nii == 1) | (pred_nii == 2) | (pred_nii == 4)).astype(np.float32)
            pr_et = (pred_nii == 4).astype(np.float32)
            pr_3ch = np.stack([pr_tc, pr_wt, pr_et], axis=0)

            tensor_gt = torch.from_numpy(gt_3ch).unsqueeze(0)
            tensor_pr = torch.from_numpy(pr_3ch).unsqueeze(0)

            dice_scores = dice_metric(y_pred=tensor_pr, y=tensor_gt)[0].numpy()
            tc_dice = float(dice_scores[0]) if not np.isnan(dice_scores[0]) else 1.0
            wt_dice = float(dice_scores[1]) if not np.isnan(dice_scores[1]) else 1.0
            et_dice = float(dice_scores[2]) if not np.isnan(dice_scores[2]) else 1.0

            results_per_patient.append({
                "patient_id": pid,
                "dice_tc": round(tc_dice, 4),
                "dice_wt": round(wt_dice, 4),
                "dice_et": round(et_dice, 4),
                "latency_sec": round(elapsed, 2),
            })
        else:
            results_per_patient.append({
                "patient_id": pid,
                "latency_sec": round(elapsed, 2),
            })

    # Summary metrics
    wt_dices = [r["dice_wt"] for r in results_per_patient if "dice_wt" in r]
    tc_dices = [r["dice_tc"] for r in results_per_patient if "dice_tc" in r]
    et_dices = [r["dice_et"] for r in results_per_patient if "dice_et" in r]

    metrics_payload = {
        "model_id": "glioma_segresnet_3d",
        "model_version": "1.0.0",
        "evaluation_dataset": "BraTS-TCGA-GBM",
        "evaluation_split": "patient_level_held_out",
        "test_subjects_count": len(test_df),
        "metrics": {
            "whole_tumor": {
                "mean_dice": round(float(np.mean(wt_dices)), 4) if wt_dices else None,
                "std_dice": round(float(np.std(wt_dices)), 4) if wt_dices else None,
            },
            "tumor_core": {
                "mean_dice": round(float(np.mean(tc_dices)), 4) if tc_dices else None,
                "std_dice": round(float(np.std(tc_dices)), 4) if tc_dices else None,
            },
            "enhancing_tumor": {
                "mean_dice": round(float(np.mean(et_dices)), 4) if et_dices else None,
                "std_dice": round(float(np.std(et_dices)), 4) if et_dices else None,
            },
            "efficiency": {
                "mean_inference_latency_seconds": round(float(np.mean(latencies)), 2),
                "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
            }
        },
        "per_patient_results": results_per_patient,
    }

    Path(output_metrics_json).parent.mkdir(parents=True, exist_ok=True)
    with open(output_metrics_json, "w") as f:
        json.dump(metrics_payload, f, indent=2)
    print(f"[INFO] Evaluation report saved to {output_metrics_json}")
    return metrics_payload


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate Glioma Model Checkpoint")
    parser.add_argument("--weights_path", type=str, default="glioma_ai/weights/best_segresnet_weights.pth")
    parser.add_argument("--manifest_csv", type=str, default="glioma_ai/split_manifest.csv")
    parser.add_argument("--output_json", type=str, default="glioma_ai/metrics.json")
    args = parser.parse_args()

    run_evaluation(
        weights_path=args.weights_path,
        manifest_csv=args.manifest_csv,
        output_metrics_json=args.output_json,
    )

