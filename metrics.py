"""
Evaluation metrics for binary change detection.

Standard metrics: F1, IoU, Precision, Recall, Accuracy, Kappa
"""

import numpy as np
import torch
from typing import Dict


class Metrics:
    """
    Accumulator for binary change detection metrics.

    Tracks confusion matrix: TP, TN, FP, FN
    """

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self.reset()

    def reset(self):
        """Reset counters."""
        self.tp = 0
        self.tn = 0
        self.fp = 0
        self.fn = 0

    @torch.no_grad()
    def update(self, pred: torch.Tensor, target: torch.Tensor):
        """
        Update with batch of predictions.

        Args:
            pred: (B, 1, H, W) logits
            target: (B, H, W) binary mask
        """
        # Get binary predictions
        pred_binary = (torch.sigmoid(pred.squeeze(1)) > self.threshold).float()
        target_binary = target.float()

        # Update confusion matrix
        self.tp += ((pred_binary == 1) & (target_binary == 1)).sum().item()
        self.tn += ((pred_binary == 0) & (target_binary == 0)).sum().item()
        self.fp += ((pred_binary == 1) & (target_binary == 0)).sum().item()
        self.fn += ((pred_binary == 0) & (target_binary == 1)).sum().item()

    def compute(self) -> Dict[str, float]:
        """Compute all metrics."""
        eps = 1e-7

        precision = self.tp / (self.tp + self.fp + eps)
        recall = self.tp / (self.tp + self.fn + eps)
        f1 = 2 * precision * recall / (precision + recall + eps)

        # Foreground IoU (change class)
        iou_fg = self.tp / (self.tp + self.fp + self.fn + eps)

        # Background IoU (no-change class)
        iou_bg = self.tn / (self.tn + self.fp + self.fn + eps)

        # Mean IoU
        miou = (iou_fg + iou_bg) / 2

        total = self.tp + self.tn + self.fp + self.fn
        accuracy = (self.tp + self.tn) / (total + eps)

        # Kappa coefficient
        po = accuracy
        pe = (
            ((self.tp + self.fn) * (self.tp + self.fp) +
             (self.tn + self.fp) * (self.tn + self.fn))
            / (total ** 2 + eps)
        )
        kappa = (po - pe) / (1 - pe + eps)

        return {
            "f1": f1,
            "iou": iou_fg,
            "iou_bg": iou_bg,
            "miou": miou,
            "precision": precision,
            "recall": recall,
            "accuracy": accuracy,
            "kappa": kappa,
        }

    def __str__(self) -> str:
        m = self.compute()
        return (
            f"F1: {m['f1']:.4f} | IoU: {m['iou']:.4f} | "
            f"Prec: {m['precision']:.4f} | Rec: {m['recall']:.4f}"
        )
