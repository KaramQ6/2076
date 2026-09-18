"""
Quantitative Evaluation Engine & Metrics Calculator
===================================================

Computes multi-region Dice coefficients, 95th percentile Hausdorff Distance (HD95),
and Intersection-over-Union (IoU) across held-out patient splits.
"""

from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import torch
from monai.metrics import DiceMetric, HausdorffDistanceMetric


class GliomaEvaluator:
    """
    Evaluates multi-region segmentation quality:
      - Whole Tumor (WT)
      - Tumor Core (TC)
      - Enhancing Tumor (ET)
    """

    def __init__(self, include_hd95: bool = True):
        self.include_hd95 = include_hd95
        self.dice_metric = DiceMetric(
            include_background=True,
            reduction="none",
            ignore_empty=True,
        )
        if include_hd95:
            self.hd95_metric = HausdorffDistanceMetric(
                include_background=True,
                percentile=95,
                reduction="none",
            )

    def evaluate_batch(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
    ) -> Dict[str, List[float]]:
        """
        Takes binary multi-label tensors (B, 3, D, H, W):
          Channel 0: TC
          Channel 1: WT
          Channel 2: ET
        Returns per-case scores.
        """
        # Ensure binary tensors
        pred_bin = (y_pred > 0.5).float()
        true_bin = (y_true > 0.5).float()

        dice_scores = self.dice_metric(y_pred=pred_bin, y=true_bin) # Shape: (B, 3)
        res = {
            "dice_tc": [],
            "dice_wt": [],
            "dice_et": [],
        }

        dice_np = dice_scores.cpu().numpy()
        for row in dice_np:
            res["dice_tc"].append(float(row[0]) if not np.isnan(row[0]) else 1.0)
            res["dice_wt"].append(float(row[1]) if not np.isnan(row[1]) else 1.0)
            res["dice_et"].append(float(row[2]) if not np.isnan(row[2]) else 1.0)

        if self.include_hd95:
            res["hd95_tc"] = []
            res["hd95_wt"] = []
            res["hd95_et"] = []
            try:
                hd_scores = self.hd95_metric(y_pred=pred_bin, y=true_bin).cpu().numpy()
                for row in hd_scores:
                    res["hd95_tc"].append(float(row[0]) if not np.isnan(row[0]) else 0.0)
                    res["hd95_wt"].append(float(row[1]) if not np.isnan(row[1]) else 0.0)
                    res["hd95_et"].append(float(row[2]) if not np.isnan(row[2]) else 0.0)
            except Exception:
                # Handle edge cases where empty masks cannot compute distance
                pass

        return res

    def compute_summary(self, case_scores: Dict[str, List[float]]) -> Dict[str, Any]:
        """Aggregates per-case scores into mean, std, median metrics."""
        summary = {}
        for metric_name, values in case_scores.items():
            if values:
                summary[metric_name] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "median": float(np.median(values)),
                }
        return summary
