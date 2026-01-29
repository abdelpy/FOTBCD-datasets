"""
Loss functions for binary change detection.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def lovasz_grad(gt_sorted: torch.Tensor) -> torch.Tensor:
    """Compute gradient of the Lovasz extension w.r.t sorted errors."""
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1.0 - intersection / union
    if p > 1:
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


def lovasz_hinge_flat(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Binary Lovasz hinge loss on flattened tensors."""
    if len(labels) == 0:
        return logits.sum() * 0.0
    signs = 2.0 * labels.float() - 1.0
    errors = 1.0 - logits * signs
    errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
    perm = perm.data
    gt_sorted = labels[perm]
    grad = lovasz_grad(gt_sorted)
    loss = torch.dot(F.relu(errors_sorted), grad)
    return loss


class LovaszHingeLoss(nn.Module):
    """
    Lovasz-Hinge loss for binary segmentation.
    """

    def __init__(self, per_image: bool = True):
        super().__init__()
        self.per_image = per_image

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: (B, 1, H, W) logits
            target: (B, H, W) binary mask
        """
        pred = pred.squeeze(1)  # (B, H, W)
        target = target.float()

        if self.per_image:
            losses = []
            for p, t in zip(pred, target):
                losses.append(lovasz_hinge_flat(p.view(-1), t.view(-1)))
            return torch.stack(losses).mean()
        else:
            return lovasz_hinge_flat(pred.view(-1), target.view(-1))


class DiceLoss(nn.Module):
    """Binary Dice loss (per-sample, then averaged)."""

    def __init__(self, smooth: float = 0.1):  # Reduced from 1.0 for sharper gradients
        super().__init__()
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: (B, 1, H, W) logits
            target: (B, H, W) binary mask

        Returns:
            loss: scalar
        """
        pred = torch.sigmoid(pred)  # (B, 1, H, W)
        if target.dim() == 3:
            target = target.unsqueeze(1)  # (B, 1, H, W)
        target = target.float()

        # Per-sample dice (avoids flattening to giant tensor)
        pred_flat = pred.view(pred.size(0), -1)  # (B, H*W)
        target_flat = target.view(target.size(0), -1)  # (B, H*W)

        intersection = (pred_flat * target_flat).sum(dim=1)
        pred_sum = pred_flat.sum(dim=1)
        target_sum = target_flat.sum(dim=1)

        dice = (2.0 * intersection + self.smooth) / (pred_sum + target_sum + self.smooth)

        return 1.0 - dice.mean()


class FocalLoss(nn.Module):
    """Focal loss for class imbalance."""

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: (B, 1, H, W) logits
            target: (B, H, W) binary mask
        """
        pred = pred.squeeze(1)  # (B, H, W)
        target = target.float()

        bce = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
        p_t = torch.sigmoid(pred) * target + (1 - torch.sigmoid(pred)) * (1 - target)
        focal_weight = (1 - p_t) ** self.gamma
        loss = self.alpha * focal_weight * bce

        return loss.mean()


class BCEDiceLoss(nn.Module):
    """BCE + Dice combined loss."""

    def __init__(self, bce_weight: float = 1.0, dice_weight: float = 1.0):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.dice = DiceLoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: (B, 1, H, W) logits
            target: (B, H, W) binary mask

        Returns:
            loss: scalar
        """
        pred_squeezed = pred.squeeze(1)  # (B, H, W)
        target = target.float()

        bce = F.binary_cross_entropy_with_logits(pred_squeezed, target)
        dice = self.dice(pred, target)

        return self.bce_weight * bce + self.dice_weight * dice


class FocalDiceLoss(nn.Module):
    """Focal + Dice combined loss."""

    def __init__(self, focal_weight: float = 1.0, dice_weight: float = 1.0, gamma: float = 2.0):
        super().__init__()
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
        self.focal = FocalLoss(gamma=gamma)
        self.dice = DiceLoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        focal = self.focal(pred, target)
        dice = self.dice(pred, target)
        return self.focal_weight * focal + self.dice_weight * dice


class BoundaryLoss(nn.Module):
    """
    Boundary loss using Laplacian edge detection.
    Laplacian catches edges better than Sobel alone.
    """

    def __init__(self):
        super().__init__()
        # Laplacian kernel (sharper than Sobel)
        laplacian = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer("laplacian", laplacian)

    def get_edges(self, x: torch.Tensor) -> torch.Tensor:
        """Extract edges using Laplacian operator."""
        if x.dim() == 3:
            x = x.unsqueeze(1)
        laplacian = self.laplacian.to(device=x.device, dtype=x.dtype)
        edges = F.conv2d(x, laplacian, padding=1)
        return torch.abs(edges)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: (B, 1, H, W) logits
            target: (B, H, W) binary mask
        """
        pred_prob = torch.sigmoid(pred)
        target = target.float().unsqueeze(1)

        pred_edges = self.get_edges(pred_prob)
        target_edges = self.get_edges(target)

        diff = torch.abs(pred_edges - target_edges)
        return diff.mean(dim=[1, 2, 3]).mean()


class BoundaryAwareBCELoss(nn.Module):
    """
    BCE loss with hard boundary mining.
    Pixels near edges get higher weight.
    """

    def __init__(self, boundary_weight: float = 3.0, dilation: int = 3):
        super().__init__()
        self.boundary_weight = boundary_weight
        self.dilation = dilation
        # Dilation kernel for boundary extraction
        kernel_size = 2 * dilation + 1
        kernel = torch.ones(1, 1, kernel_size, kernel_size)
        self.register_buffer("dilate_kernel", kernel)

    def get_boundary_mask(self, target: torch.Tensor) -> torch.Tensor:
        """Get boundary region mask via morphological dilation - erosion."""
        if target.dim() == 3:
            target = target.unsqueeze(1)
        target = target.float()
        kernel = self.dilate_kernel.to(device=target.device, dtype=target.dtype)

        # Dilate and erode
        dilated = F.conv2d(target, kernel, padding=self.dilation)
        dilated = (dilated > 0).float()
        eroded = F.conv2d(target, kernel, padding=self.dilation)
        eroded = (eroded >= kernel.sum()).float()

        # Boundary = dilated - eroded
        boundary = dilated - eroded
        return boundary.squeeze(1)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: (B, 1, H, W) logits
            target: (B, H, W) binary mask
        """
        pred = pred.squeeze(1)
        target = target.float()

        # Get boundary mask
        boundary_mask = self.get_boundary_mask(target)

        # Weight map: higher weight on boundaries
        weights = torch.ones_like(target)
        weights = weights + (self.boundary_weight - 1.0) * boundary_mask

        # Weighted BCE
        bce = F.binary_cross_entropy_with_logits(pred, target, weight=weights, reduction='none')
        return bce.mean(dim=[1, 2]).mean()


class BCEDiceBoundaryLoss(nn.Module):
    """BCE + Dice + Boundary combined loss for crisp edges."""

    def __init__(self, bce_weight: float = 1.0, dice_weight: float = 1.0, boundary_weight: float = 1.0):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.boundary_weight = boundary_weight
        self.dice = DiceLoss()
        self.boundary = BoundaryLoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_squeezed = pred.squeeze(1)
        target = target.float()

        bce = F.binary_cross_entropy_with_logits(pred_squeezed, target, reduction='none')
        bce = bce.mean(dim=[1, 2]).mean()

        dice = self.dice(pred, target)
        boundary = self.boundary(pred, target)

        return self.bce_weight * bce + self.dice_weight * dice + self.boundary_weight * boundary


class SharpLoss(nn.Module):
    """
    The good stuff: Lovasz + Boundary-aware BCE + Laplacian boundary.
    Designed for crisp masks.
    """

    def __init__(
        self,
        lovasz_weight: float = 1.0,
        bce_weight: float = 1.0,
        boundary_weight: float = 1.5,
        edge_mining_weight: float = 3.0,
    ):
        super().__init__()
        self.lovasz_weight = lovasz_weight
        self.bce_weight = bce_weight
        self.boundary_weight = boundary_weight

        self.lovasz = LovaszHingeLoss(per_image=True)
        self.boundary_bce = BoundaryAwareBCELoss(boundary_weight=edge_mining_weight)
        self.boundary = BoundaryLoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: (B, 1, H, W) logits
            target: (B, H, W) binary mask
        """
        lovasz = self.lovasz(pred, target)
        bce = self.boundary_bce(pred, target)
        boundary = self.boundary(pred, target)

        return self.lovasz_weight * lovasz + self.bce_weight * bce + self.boundary_weight * boundary


# Loss Factory

LOSSES = {
    "bce": lambda cfg: nn.BCEWithLogitsLoss(),
    "dice": lambda cfg: DiceLoss(),
    "bce_dice": lambda cfg: BCEDiceLoss(cfg.BCE_WEIGHT, cfg.DICE_WEIGHT),
    "focal": lambda cfg: FocalLoss(gamma=cfg.FOCAL_GAMMA),
    "focal_dice": lambda cfg: FocalDiceLoss(gamma=cfg.FOCAL_GAMMA),
    "bce_dice_boundary": lambda cfg: BCEDiceBoundaryLoss(
        cfg.BCE_WEIGHT, cfg.DICE_WEIGHT, getattr(cfg, "BOUNDARY_WEIGHT", 1.0)
    ),
    "lovasz": lambda _: LovaszHingeLoss(per_image=True),
    "sharp": lambda cfg: SharpLoss(
        lovasz_weight=getattr(cfg, "LOVASZ_WEIGHT", 1.0),
        bce_weight=cfg.BCE_WEIGHT,
        boundary_weight=getattr(cfg, "BOUNDARY_WEIGHT", 1.5),
        edge_mining_weight=getattr(cfg, "EDGE_MINING_WEIGHT", 3.0),
    ),
}


def build_loss(cfg) -> nn.Module:
    """Build loss from config."""
    loss_type = cfg.LOSS_TYPE.lower()
    if loss_type not in LOSSES:
        raise ValueError(f"Unknown loss: {loss_type}. Available: {list(LOSSES.keys())}")
    return LOSSES[loss_type](cfg)
