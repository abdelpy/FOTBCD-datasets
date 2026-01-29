"""
Training script for Change Detection Benchmark.

Step-based logging for fair cross-dataset comparison.

Usage:
    python train.py
    DATASET=levir python train.py
    DATASET=fotbcd python train.py
"""

import os
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
from PIL import Image

from config import CFG, DATASET_PRESETS
from model import build_model
from dataset import get_dataloaders, get_dataset
from losses import build_loss
from metrics import Metrics


def denormalize(tensor, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    """Denormalize image tensor for visualization."""
    mean = torch.tensor(mean, device=tensor.device).view(1, 3, 1, 1)
    std = torch.tensor(std, device=tensor.device).view(1, 3, 1, 1)
    return (tensor * std + mean).clamp(0, 1)


def save_visualizations(model, samples, device, save_path, step, mean_iou):
    """Save visualization grid at given step."""
    model.eval()
    vis_dir = save_path / "visualizations"
    vis_dir.mkdir(exist_ok=True)

    rows = []
    with torch.no_grad():
        for sample in samples:
            before = sample["A"].unsqueeze(0).to(device)
            after = sample["B"].unsqueeze(0).to(device)
            mask_gt = sample["mask"].unsqueeze(0).to(device)

            with torch.amp.autocast("cuda", enabled=True):
                pred_logits = model(before, after)
            pred_mask = (torch.sigmoid(pred_logits) > 0.5).float()

            before_img = denormalize(before)[0].permute(1, 2, 0).cpu().numpy()
            after_img = denormalize(after)[0].permute(1, 2, 0).cpu().numpy()
            mask_gt_img = mask_gt[0].cpu().numpy()
            pred_mask_img = pred_mask[0, 0].cpu().numpy()

            h, w = before_img.shape[:2]
            row = np.zeros((h, w * 4, 3), dtype=np.uint8)
            row[:, 0:w] = (before_img * 255).astype(np.uint8)
            row[:, w:2*w] = (after_img * 255).astype(np.uint8)
            row[:, 2*w:3*w] = (np.stack([mask_gt_img]*3, axis=-1) * 255).astype(np.uint8)
            row[:, 3*w:4*w] = (np.stack([pred_mask_img]*3, axis=-1) * 255).astype(np.uint8)
            rows.append(row)

    grid = np.vstack(rows)
    img = Image.fromarray(grid)
    img.save(vis_dir / f"step_{step:06d}_iou_{mean_iou:.3f}.png")

    model.train()


def get_fixed_samples(dataset, n_samples=8, seed=42):
    """Get fixed random samples for consistent visualization."""
    random.seed(seed)
    indices = random.sample(range(len(dataset)), min(n_samples, len(dataset)))
    return [dataset[i] for i in indices], indices


def evaluate_fixed_samples(model, samples, device):
    """Evaluate IoU on fixed samples."""
    model.eval()
    ious = []

    with torch.no_grad():
        for sample in samples:
            before = sample["A"].unsqueeze(0).to(device)
            after = sample["B"].unsqueeze(0).to(device)
            mask_gt = sample["mask"].unsqueeze(0).to(device)

            with torch.amp.autocast("cuda", enabled=True):
                pred_logits = model(before, after)
            pred_mask = (torch.sigmoid(pred_logits) > 0.5).float()

            intersection = (pred_mask * mask_gt).sum()
            union = pred_mask.sum() + mask_gt.sum() - intersection
            iou = (intersection / (union + 1e-6)).item()
            ious.append(iou)

    model.train()
    return {"mean_iou": np.mean(ious), "ious": ious}


@torch.no_grad()
def validate(model, loader, criterion, device, desc="Val"):
    """Validate model."""
    model.eval()
    metrics = Metrics()
    total_loss = 0

    for batch in tqdm(loader, desc=desc):
        before = batch["A"].to(device)
        after = batch["B"].to(device)
        mask = batch["mask"].to(device)

        with torch.amp.autocast("cuda", enabled=True):
            pred = model(before, after)
            loss = criterion(pred, mask)

        metrics.update(pred, mask)
        total_loss += loss.item()

    m = metrics.compute()
    m["loss"] = total_loss / len(loader)
    return m


def load_cross_eval_loaders(current_dataset, img_size, batch_size, num_workers):
    """Load validation loaders for other datasets (cross-domain evaluation)."""
    cross_loaders = {}

    for name, preset in DATASET_PRESETS.items():
        if name == current_dataset:
            continue

        try:
            crop_size = preset.get("crop_size", img_size)
            original_size = preset.get("original_size", img_size)

            dataset = get_dataset(
                name, preset["data_root"], split="test", img_size=img_size,
                augment=False, crop_size=crop_size, original_size=original_size
            )

            loader = torch.utils.data.DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True,
            )
            cross_loaders[name] = loader
            print(f"  Cross-eval: {name} ({len(dataset)} patches)")

        except Exception as e:
            print(f"  Cross-eval: {name} - SKIPPED ({e})")

    return cross_loaders


def main():
    device = CFG.DEVICE
    print(f"Device: {device}")
    print(f"Dataset: {CFG.DATASET}")
    print(f"Encoder: {CFG.ENCODER}")

    # Create output directory
    save_dir = Path(CFG.SAVE_DIR) / CFG.EXPERIMENT_NAME
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving to: {save_dir}")

    # Single TensorBoard writer - all metrics in one place
    writer = SummaryWriter(log_dir=save_dir / "tensorboard")

    # Data
    train_loader, val_loader, test_loader = get_dataloaders(
        name=CFG.DATASET,
        root=CFG.DATA_ROOT,
        batch_size=CFG.BATCH_SIZE,
        img_size=CFG.IMG_SIZE,
        num_workers=CFG.NUM_WORKERS,
        quick_test=getattr(CFG, "QUICK_TEST", False),
        quick_test_train_samples=getattr(CFG, "QUICK_TEST_TRAIN_SAMPLES", 100),
        quick_test_val_samples=getattr(CFG, "QUICK_TEST_VAL_SAMPLES", 50),
        crop_size=CFG.CROP_SIZE,
        original_size=CFG.ORIGINAL_SIZE,
    )
    print(f"Train: {len(train_loader.dataset)} patches from {CFG.DATA_ROOT}")
    print(f"Val: {len(val_loader.dataset)} patches from {CFG.DATA_ROOT}")
    print(f"Steps per epoch: {len(train_loader)}")

    # Compute and log patch statistics (for fairness analysis)
    print("\nPatch statistics (sampled):")
    if hasattr(train_loader.dataset, 'compute_statistics'):
        train_stats = train_loader.dataset.compute_statistics(num_samples=2000)
        print(f"  Train: {train_stats['empty_ratio']*100:.1f}% empty, "
              f"avg change={train_stats['avg_change_ratio']*100:.2f}%, "
              f"median={train_stats['median_change_ratio']*100:.2f}%")
    if hasattr(val_loader.dataset, 'compute_statistics'):
        val_stats = val_loader.dataset.compute_statistics(num_samples=1000)
        print(f"  Val:   {val_stats['empty_ratio']*100:.1f}% empty, "
              f"avg change={val_stats['avg_change_ratio']*100:.2f}%, "
              f"median={val_stats['median_change_ratio']*100:.2f}%")

    # Get fixed samples for visualization
    n_vis_samples = getattr(CFG, "N_VIS_SAMPLES", 8)
    fixed_samples, fixed_indices = get_fixed_samples(val_loader.dataset, n_samples=n_vis_samples)
    print(f"Fixed visualization samples: {fixed_indices}")

    # Load cross-eval datasets (other datasets for generalization testing)
    cross_eval_every = getattr(CFG, "CROSS_EVAL_EVERY_STEPS", 0)
    cross_loaders = {}
    if cross_eval_every > 0:
        print("\nLoading cross-evaluation datasets...")
        cross_loaders = load_cross_eval_loaders(
            CFG.DATASET, CFG.IMG_SIZE, CFG.BATCH_SIZE, CFG.NUM_WORKERS
        )

    # Model
    model = build_model(CFG)
    model = model.to(device)

    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {n_params:,} total, {n_trainable:,} trainable")

    # Loss & optimizer
    criterion = build_loss(CFG)
    optimizer = optim.AdamW(model.parameters(), lr=CFG.LR, weight_decay=CFG.WEIGHT_DECAY)

    # Step-based scheduler (fair across datasets with different sizes)
    total_steps = getattr(CFG, "TOTAL_STEPS", 50000)
    warmup_steps = getattr(CFG, "WARMUP_STEPS", 2000)

    # Cosine annealing with linear warmup
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps  # Linear warmup
        # Cosine decay after warmup
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        return 0.01 + 0.99 * 0.5 * (1 + np.cos(np.pi * progress))  # decay to 1% of initial LR

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = torch.amp.GradScaler("cuda", enabled=CFG.USE_AMP)

    # Step-based config
    log_every = getattr(CFG, "LOG_EVERY_STEPS", 100)
    val_every = getattr(CFG, "VAL_EVERY_STEPS", 1000)
    save_every = getattr(CFG, "SAVE_EVERY_STEPS", 5000)

    # Training state
    global_step = 0
    best_iou = 0
    best_loss = float("inf")

    # Resume from checkpoint if exists
    resume_path = save_dir / "latest.pth"
    if resume_path.exists():
        print(f"Resuming from {resume_path}")
        ckpt = torch.load(resume_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        scaler.load_state_dict(ckpt["scaler"])
        global_step = ckpt.get("global_step", 0)
        best_iou = ckpt.get("best_iou", 0)
        best_loss = ckpt.get("best_loss", float("inf"))
        print(f"  Resumed at step {global_step}, best_iou={best_iou:.4f}")

    # Training metrics accumulator (reset every log_every steps)
    train_metrics = Metrics()
    train_loss_acc = 0
    steps_since_log = 0

    # Step-based training loop (fair comparison across datasets)
    print(f"\n{'='*50}")
    print(f"Training for {total_steps} steps (warmup: {warmup_steps} steps)")
    print("=" * 50)

    model.train()
    data_iter = iter(train_loader)
    pbar = tqdm(total=total_steps, initial=global_step, desc="Training")

    while global_step < total_steps:
        # Get next batch (cycle through dataloader)
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)

        before = batch["A"].to(device)
        after = batch["B"].to(device)
        mask = batch["mask"].to(device)

        optimizer.zero_grad()

        with torch.amp.autocast("cuda", enabled=scaler.is_enabled()):
            pred = model(before, after)
            loss = criterion(pred, mask)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        # Step-based LR scheduling (after optimizer step)
        scheduler.step()

        # Accumulate train metrics
        train_metrics.update(pred.detach(), mask)
        train_loss_acc += loss.item()
        steps_since_log += 1
        global_step += 1
        pbar.update(1)

        # Log train metrics every LOG_EVERY_STEPS
        if global_step % log_every == 0:
            m = train_metrics.compute()
            avg_loss = train_loss_acc / steps_since_log

            # TensorBoard - simple scalar names, all in one file
            writer.add_scalar("train/loss", avg_loss, global_step)
            writer.add_scalar("train/iou", m["iou"], global_step)
            writer.add_scalar("train/f1", m["f1"], global_step)
            writer.add_scalar("train/precision", m["precision"], global_step)
            writer.add_scalar("train/recall", m["recall"], global_step)
            writer.add_scalar("lr", scheduler.get_last_lr()[0], global_step)

            pbar.set_postfix(
                loss=f"{avg_loss:.4f}",
                iou=f"{m['iou']:.4f}",
                lr=f"{scheduler.get_last_lr()[0]:.2e}"
            )

            # Reset accumulators
            train_metrics = Metrics()
            train_loss_acc = 0
            steps_since_log = 0

        # Validation every VAL_EVERY_STEPS
        if global_step % val_every == 0:
            val_m = validate(model, val_loader, criterion, device)
            fixed_m = evaluate_fixed_samples(model, fixed_samples, device)

            # TensorBoard - val metrics
            writer.add_scalar("val/loss", val_m["loss"], global_step)
            writer.add_scalar("val/iou", val_m["iou"], global_step)
            writer.add_scalar("val/f1", val_m["f1"], global_step)
            writer.add_scalar("val/miou", val_m["miou"], global_step)
            writer.add_scalar("val/iou_bg", val_m["iou_bg"], global_step)
            writer.add_scalar("val/precision", val_m["precision"], global_step)
            writer.add_scalar("val/recall", val_m["recall"], global_step)
            writer.add_scalar("fixed/mean_iou", fixed_m["mean_iou"], global_step)

            print(f"\n[Step {global_step}] Val - Loss: {val_m['loss']:.4f}, "
                  f"IoU: {val_m['iou']:.4f}, F1: {val_m['f1']:.4f}, mIoU: {val_m['miou']:.4f}")
            print(f"[Step {global_step}] Fixed samples IoU: {fixed_m['mean_iou']:.4f}")

            # Save visualizations
            save_visualizations(model, fixed_samples, device, save_dir, global_step, fixed_m["mean_iou"])

            # Save best IoU
            if val_m["iou"] > best_iou:
                best_iou = val_m["iou"]
                torch.save(model.state_dict(), save_dir / "best_iou.pth")
                print(f"  -> New best IoU: {best_iou:.4f}")

            # Save best loss
            if val_m["loss"] < best_loss:
                best_loss = val_m["loss"]
                torch.save(model.state_dict(), save_dir / "best_loss.pth")
                print(f"  -> New best loss: {best_loss:.4f}")

            model.train()

        # Cross-dataset evaluation every CROSS_EVAL_EVERY_STEPS
        if cross_eval_every > 0 and global_step % cross_eval_every == 0 and cross_loaders:
            print(f"\n[Step {global_step}] Cross-dataset evaluation...")
            for ds_name, ds_loader in cross_loaders.items():
                cross_m = validate(model, ds_loader, criterion, device, desc=f"Cross-{ds_name}")

                # TensorBoard - cross-eval metrics (iou, precision, recall, OA, f1)
                writer.add_scalar(f"cross/{ds_name}/iou", cross_m["iou"], global_step)
                writer.add_scalar(f"cross/{ds_name}/precision", cross_m["precision"], global_step)
                writer.add_scalar(f"cross/{ds_name}/recall", cross_m["recall"], global_step)
                writer.add_scalar(f"cross/{ds_name}/oa", cross_m["accuracy"], global_step)
                writer.add_scalar(f"cross/{ds_name}/f1", cross_m["f1"], global_step)

                print(f"  {ds_name}: IoU={cross_m['iou']:.4f}, P={cross_m['precision']:.4f}, "
                      f"R={cross_m['recall']:.4f}, OA={cross_m['accuracy']:.4f}, F1={cross_m['f1']:.4f}")

            model.train()

        # Save checkpoint every SAVE_EVERY_STEPS
        if global_step % save_every == 0:
            torch.save({
                "global_step": global_step,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "best_iou": best_iou,
                "best_loss": best_loss,
            }, save_dir / f"step_{global_step}.pth")

            # Also save as latest
            torch.save({
                "global_step": global_step,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "best_iou": best_iou,
                "best_loss": best_loss,
            }, save_dir / "latest.pth")

    pbar.close()

    # Save final
    torch.save(model.state_dict(), save_dir / "final.pth")

    # Test evaluation
    if test_loader:
        print("\n" + "=" * 50)
        print("Test Evaluation")
        print("=" * 50)
        test_m = validate(model, test_loader, criterion, device)
        print(f"Test - IoU: {test_m['iou']:.4f}, F1: {test_m['f1']:.4f}, mIoU: {test_m['miou']:.4f}")

        writer.add_scalar("test/iou", test_m["iou"], global_step)
        writer.add_scalar("test/f1", test_m["f1"], global_step)
        writer.add_scalar("test/miou", test_m["miou"], global_step)

    writer.close()
    print(f"\nDone. Best IoU: {best_iou:.4f}, Best Loss: {best_loss:.4f}")
    print(f"Total steps: {global_step}")


if __name__ == "__main__":
    main()
