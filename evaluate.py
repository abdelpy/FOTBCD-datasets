"""
Evaluation script for Change Detection Benchmark.

Evaluates best checkpoint from each experiment against all datasets.

Usage:
    python evaluate.py                              # Auto-discover experiments
    python evaluate.py --checkpoints_dir ./checkpoints
    python evaluate.py --batch_size 32
"""

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import CFG, DATASET_PRESETS
from model import build_model
from dataset import get_dataset
from metrics import Metrics


def load_model(checkpoint_path: str, device: str = "cuda"):
    """Load model from checkpoint."""
    CFG.PRETRAINED = False
    model = build_model(CFG)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if "model" in checkpoint:
        model.load_state_dict(checkpoint["model"])
    else:
        model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()
    return model


@torch.no_grad()
def evaluate(model, loader, device):
    """Evaluate model on dataloader."""
    metrics = Metrics()

    for batch in tqdm(loader, desc="Evaluating", leave=False):
        before = batch["A"].to(device)
        after = batch["B"].to(device)
        mask = batch["mask"].to(device)

        with torch.amp.autocast("cuda"):
            pred = model(before, after)

        metrics.update(pred, mask)

    return metrics.compute()


def discover_experiments(checkpoints_dir: Path):
    """Find all experiment directories with best checkpoints."""
    experiments = {}

    for exp_dir in checkpoints_dir.iterdir():
        if not exp_dir.is_dir():
            continue

        # Look for best.pth
        best_ckpt = exp_dir / "best.pth"
        if not best_ckpt.exists():
            continue

        # Folder name must match a DATASET_PRESETS key
        folder_name = exp_dir.name.lower()
        if folder_name not in DATASET_PRESETS:
            print(f"  Warning: Unknown checkpoint folder '{folder_name}', skipping")
            continue

        experiments[folder_name] = {
            "dir": exp_dir,
            "checkpoint": best_ckpt,
            "train_dataset": folder_name,
        }

    return experiments


def get_test_loader(dataset_name: str, batch_size: int, num_workers: int = 4):
    """Get test dataloader for a dataset."""
    preset = DATASET_PRESETS[dataset_name]

    crop_size = preset.get("crop_size", 256)
    original_size = preset.get("original_size", 256)
    img_size = preset.get("img_size", 256)

    dataset = get_dataset(
        dataset_name,
        preset["data_root"],
        split="test",
        img_size=img_size,
        augment=False,
        crop_size=crop_size,
        original_size=original_size,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return loader, len(dataset)


def format_results_table(results: dict) -> str:
    """Format results as a table string."""
    lines = []
    all_datasets = list(DATASET_PRESETS.keys())

    header = f"{'Experiment':<35} | {'Eval Dataset':<12} | {'F1':>7} | {'IoU':>7} | {'Prec':>7} | {'Rec':>7} | {'OA':>7} | {'Kappa':>7}"
    sep = "-" * len(header)

    lines.append("\n" + "=" * len(header))
    lines.append("CROSS-DATASET EVALUATION RESULTS")
    lines.append("=" * len(header))
    lines.append(header)
    lines.append(sep)

    for exp_name, exp_results in results.items():
        train_ds = exp_results["train_dataset"]

        for eval_ds in all_datasets:
            if eval_ds not in exp_results["metrics"]:
                continue

            m = exp_results["metrics"][eval_ds]
            ds_label = f"{eval_ds}*" if eval_ds == train_ds else eval_ds

            lines.append(f"{exp_name:<35} | {ds_label:<12} | {m['f1']:>7.4f} | {m['iou']:>7.4f} | "
                         f"{m['precision']:>7.4f} | {m['recall']:>7.4f} | {m['accuracy']:>7.4f} | {m['kappa']:>7.4f}")

        lines.append(sep)

    lines.append("\n* = training dataset (self-evaluation on test split)")
    return "\n".join(lines)


def format_metric_matrix(results: dict, metric_key: str) -> str:
    """Format a single metric as a matrix string."""
    lines = []
    all_datasets = list(DATASET_PRESETS.keys())

    header = f"{'Trained On':<15}"
    for ds in all_datasets:
        header += f" | {ds:>10}"
    lines.append(header)
    lines.append("-" * len(header))

    for exp_results in results.values():
        train_ds = exp_results["train_dataset"]
        row = f"{train_ds:<15}"

        for eval_ds in all_datasets:
            if eval_ds in exp_results["metrics"]:
                val = exp_results["metrics"][eval_ds][metric_key]
                row += f" | {val:>10.4f}"
            else:
                row += f" | {'N/A':>10}"

        lines.append(row)

    return "\n".join(lines)


def format_summary_matrices(results: dict) -> str:
    """Format all metrics as matrices string."""
    lines = []
    metrics = [
        ("F1 Score", "f1"),
        ("IoU", "iou"),
        ("Precision", "precision"),
        ("Recall", "recall"),
        ("Overall Accuracy", "accuracy"),
        ("Kappa", "kappa"),
    ]

    for metric_name, metric_key in metrics:
        lines.append("\n" + "=" * 80)
        lines.append(f"{metric_name.upper()} MATRIX (rows=trained on, cols=evaluated on)")
        lines.append("=" * 80)
        lines.append(format_metric_matrix(results, metric_key))

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints_dir", type=str, default="./checkpoints",
                        help="Directory containing checkpoints/{dataset}/best.pth")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for evaluation")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--output", type=str, default="evaluation_results.txt",
                        help="Output file for results")
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Batch size: {args.batch_size}")

    checkpoints_dir = Path(args.checkpoints_dir)
    if not checkpoints_dir.exists():
        print(f"Error: Checkpoints directory not found: {checkpoints_dir}")
        return

    # Discover experiments
    experiments = discover_experiments(checkpoints_dir)
    if not experiments:
        print(f"No experiments found in {checkpoints_dir}")
        return

    print(f"\nFound {len(experiments)} experiments:")
    for exp_name, exp_info in experiments.items():
        print(f"  - {exp_name} (trained on: {exp_info['train_dataset']})")

    # Load test loaders for all datasets
    print("\nLoading test datasets...")
    test_loaders = {}
    for ds_name in DATASET_PRESETS.keys():
        try:
            loader, n_samples = get_test_loader(ds_name, args.batch_size, args.num_workers)
            test_loaders[ds_name] = loader
            print(f"  {ds_name}: {n_samples} samples")
        except Exception as e:
            print(f"  {ds_name}: SKIPPED ({e})")

    # Evaluate each experiment against all datasets
    print("\n" + "=" * 60)
    print("EVALUATING...")
    print("=" * 60)

    results = {}

    for exp_name, exp_info in experiments.items():
        print(f"\n{'='*60}")
        print(f"MODEL: {exp_name} (trained on {exp_info['train_dataset']})")
        print(f"Checkpoint: {exp_info['checkpoint']}")
        print("="*60)

        try:
            model = load_model(str(exp_info["checkpoint"]), device)
        except Exception as e:
            print(f"  Error loading model: {e}")
            continue

        results[exp_name] = {
            "train_dataset": exp_info["train_dataset"],
            "metrics": {}
        }

        for ds_name, loader in test_loaders.items():
            is_self = "(SELF)" if ds_name == exp_info["train_dataset"] else ""

            try:
                m = evaluate(model, loader, device)
                results[exp_name]["metrics"][ds_name] = m
                print(f"  {exp_name} -> {ds_name} {is_self}: IoU={m['iou']:.4f}, F1={m['f1']:.4f}")
            except Exception as e:
                print(f"  {exp_name} -> {ds_name} {is_self}: ERROR: {e}")

        # Free GPU memory
        del model
        torch.cuda.empty_cache()

    # Format and print results
    table_str = format_results_table(results)
    matrices_str = format_summary_matrices(results)

    print(table_str)
    print(matrices_str)

    # Write to file
    output_path = Path(args.output)
    with open(output_path, "w") as f:
        f.write(table_str + "\n")
        f.write(matrices_str + "\n")
    print(f"\nResults saved to: {output_path.absolute()}")


if __name__ == "__main__":
    main()
