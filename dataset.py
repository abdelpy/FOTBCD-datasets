"""
Standardized Dataset Classes for Change Detection Benchmarking.

Supports multiple dataset formats with unified output:
- Binary masks: 0 = no change, 1 = change

Datasets:
- FOTBCD: Binary format with 512x512 images, crops to 256x256 (before/after/label dirs)
- LEVIR-CD, WHU-CD: Standard A/B/label format with optional cropping
"""

import os
from os import path as osp
from typing import Optional, Tuple, Dict, Any

import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader, Subset
import albumentations as A
from albumentations.pytorch import ToTensorV2


# =============================================================================
# FOTBCD Dataset (Binary format) with Patch Extraction
# =============================================================================

class FOTBCDDataset(Dataset):
    """
    FOTBCD-Binary Dataset - Binary format with patch extraction.

    Works with the converted binary format from convert_to_binary_dataset.py.
    Images are 512x512, split into 256x256 crops (4 crops per image).

    Output: Binary mask (0=no_change, 1=change)

    Expected structure (from convert_to_binary_dataset.py):
        root/
            images/
                train/
                    before/     # {id}.png
                    after/      # {id}.png
                    label/      # {id}.png (0=no_change, 255=change)
                val/
                    before/
                    after/
                    label/
                test/
                    before/
                    after/
                    label/
            metadata/           # optional
                train.json
                val.json
                test.json
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        img_size: int = 256,
        augment: bool = False,
        crop_size: int = 256,
        original_size: int = 512,
        **kwargs,
    ):
        self.root = root
        self.split = split
        self.img_size = img_size
        self.augment = augment
        self.crop_size = crop_size
        self.original_size = original_size

        # Calculate crops per dimension
        self.crops_per_dim = original_size // crop_size

        # Set data directory
        self._data_dir = osp.join(self.root, "images", self.split)
        if not osp.exists(osp.join(self._data_dir, "before")):
            raise FileNotFoundError(f"Directory not found: {osp.join(self._data_dir, 'before')}")

        # Load image file list
        self._image_files = self._load_image_files()

        # Build sample list (image_idx, crop_row, crop_col)
        self.samples = self._build_crop_samples()

        # Build transforms
        self.transform = self._build_transforms()

        print(f"FOTBCDDataset {self.split}: {len(self._image_files)} images -> {len(self.samples)} crops ({self.crop_size}x{self.crop_size})")

    def _load_image_files(self) -> list:
        """Load list of image files from the before directory."""
        before_dir = osp.join(self._data_dir, "before")
        files = sorted([
            f for f in os.listdir(before_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff"))
        ])
        return files

    def _build_crop_samples(self) -> list:
        """Build list of (image_idx, crop_row, crop_col) tuples."""
        samples = []
        for img_idx in range(len(self._image_files)):
            for row in range(self.crops_per_dim):
                for col in range(self.crops_per_dim):
                    samples.append((img_idx, row, col))
        return samples

    def _build_transforms(self) -> A.Compose:
        """Build albumentations transform pipeline."""
        transforms = []

        # Resize if crop_size != img_size
        if self.crop_size != self.img_size:
            transforms.append(A.Resize(self.img_size, self.img_size))

        # Augmentations (training only)
        if self.augment:
            transforms.extend([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.RandomBrightnessContrast(p=0.3),
            ])

        # Normalize and convert to tensor
        transforms.extend([
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])

        return A.Compose(transforms, additional_targets={"image2": "image"})

    def _extract_crop(self, img: np.ndarray, row: int, col: int) -> np.ndarray:
        """Extract a crop from the image at the specified grid position."""
        y_start = row * self.crop_size
        x_start = col * self.crop_size
        return img[y_start:y_start + self.crop_size, x_start:x_start + self.crop_size]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        img_idx, row, col = self.samples[idx]
        name = self._image_files[img_idx]
        stem = osp.splitext(name)[0]

        # Load full images
        before = np.array(Image.open(osp.join(self._data_dir, "before", name)).convert("RGB"))
        after = np.array(Image.open(osp.join(self._data_dir, "after", name)).convert("RGB"))

        # Load mask (try different extensions)
        mask = None
        for ext in [".png", ".jpg", ".tif", ".tiff", ""]:
            mask_path = osp.join(self._data_dir, "label", f"{stem}{ext}")
            if not osp.exists(mask_path) and ext == "":
                mask_path = osp.join(self._data_dir, "label", name)
            if osp.exists(mask_path):
                mask = np.array(Image.open(mask_path).convert("L"))
                break

        if mask is None:
            raise FileNotFoundError(f"Mask not found for {name}")

        # Extract crops
        before_crop = self._extract_crop(before, row, col)
        after_crop = self._extract_crop(after, row, col)
        mask_crop = self._extract_crop(mask, row, col)

        # Binarize mask (255 -> 1)
        mask_crop = (mask_crop > 127).astype(np.uint8)

        # Apply transforms
        transformed = self.transform(image=before_crop, image2=after_crop, mask=mask_crop)

        return {
            "A": transformed["image"],
            "B": transformed["image2"],
            "mask": transformed["mask"].long(),
            "name": f"{stem}_r{row}_c{col}",
        }

    def compute_statistics(self, num_samples: int = 1000) -> Dict[str, float]:
        """
        Compute patch statistics for dataset analysis.

        Args:
            num_samples: Number of patches to sample (for speed). Use -1 for all.

        Returns:
            Dict with: total_patches, empty_patches, empty_ratio,
                      avg_change_ratio, median_change_ratio
        """
        import random

        indices = list(range(len(self.samples)))
        if num_samples > 0 and num_samples < len(indices):
            indices = random.sample(indices, num_samples)

        change_ratios = []
        empty_count = 0
        patch_size = self.crop_size * self.crop_size

        for idx in indices:
            img_idx, row, col = self.samples[idx]
            name = self._image_files[img_idx]
            stem = osp.splitext(name)[0]

            # Load mask
            mask = None
            for ext in [".png", ".jpg", ".tif", ".tiff", ""]:
                mask_path = osp.join(self._data_dir, "label", f"{stem}{ext}")
                if not osp.exists(mask_path) and ext == "":
                    mask_path = osp.join(self._data_dir, "label", name)
                if osp.exists(mask_path):
                    mask = np.array(Image.open(mask_path).convert("L"))
                    break

            if mask is None:
                continue

            mask_crop = self._extract_crop(mask, row, col)
            mask_crop = (mask_crop > 127).astype(np.uint8)

            change_pixels = mask_crop.sum()
            change_ratio = change_pixels / patch_size
            change_ratios.append(change_ratio)

            if change_pixels == 0:
                empty_count += 1

        return {
            "total_patches": len(self.samples),
            "sampled_patches": len(indices),
            "empty_patches": empty_count,
            "empty_ratio": empty_count / len(indices) if indices else 0,
            "avg_change_ratio": np.mean(change_ratios) if change_ratios else 0,
            "median_change_ratio": np.median(change_ratios) if change_ratios else 0,
            "std_change_ratio": np.std(change_ratios) if change_ratios else 0,
        }


# =============================================================================
# Standard A/B/label Dataset (LEVIR-CD, WHU-CD, etc.)
# =============================================================================

class ABLabelDataset(Dataset):
    """
    Standard A/B/label change detection dataset with optional cropping.

    Works with LEVIR-CD, WHU-CD, and similar datasets that use A/B/label structure.
    Supports cropping large images into smaller patches (e.g., 1024→256 for LEVIR-CD).
    For datasets already at target size (e.g., WHU-CD), set original_size=crop_size.

    Expected structure:
        root/
            train/
                A/          # before images
                B/          # after images
                label/      # change masks
            val/
            test/
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        img_size: int = 256,
        crop_size: int = 256,
        original_size: int = 256,  # Default to no cropping
        augment: bool = False,
        **kwargs,
    ):
        self.root = root
        self.split = split
        self.img_size = img_size
        self.crop_size = crop_size
        self.original_size = original_size
        self.augment = augment

        # Calculate number of crops per dimension
        self.crops_per_dim = original_size // crop_size  # 1024 // 256 = 4

        # Set data directory
        self._data_dir = osp.join(self.root, self.split)
        if not osp.exists(osp.join(self._data_dir, "A")):
            raise FileNotFoundError(f"Directory not found: {osp.join(self._data_dir, 'A')}")

        # Determine mask directory (label for LEVIR-CD+, OUT for WHU-CD)
        if osp.exists(osp.join(self._data_dir, "label")):
            self._mask_dir = "label"
        elif osp.exists(osp.join(self._data_dir, "OUT")):
            self._mask_dir = "OUT"
        else:
            raise FileNotFoundError(f"Mask directory not found (tried 'label' and 'OUT'): {self._data_dir}")

        # Load image file list
        self._image_files = self._load_image_files()

        # Build sample list (image_idx, crop_row, crop_col)
        self.samples = self._build_crop_samples()

        # Build transforms
        self.transform = self._build_transforms()

        print(f"ABLabelDataset {self.split}: {len(self._image_files)} images -> {len(self.samples)} crops ({self.crop_size}x{self.crop_size})")

    def _load_image_files(self) -> list:
        """Load list of image files."""
        before_dir = osp.join(self._data_dir, "A")
        files = sorted([
            f for f in os.listdir(before_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff"))
        ])
        return files

    def _build_crop_samples(self) -> list:
        """Build list of (image_idx, crop_row, crop_col) tuples."""
        samples = []
        for img_idx in range(len(self._image_files)):
            for row in range(self.crops_per_dim):
                for col in range(self.crops_per_dim):
                    samples.append((img_idx, row, col))
        return samples

    def _build_transforms(self) -> A.Compose:
        """Build albumentations transform pipeline."""
        transforms = []

        # Resize if crop_size != img_size
        if self.crop_size != self.img_size:
            transforms.append(A.Resize(self.img_size, self.img_size))

        # Augmentations (training only)
        if self.augment:
            transforms.extend([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.RandomBrightnessContrast(p=0.3),
            ])

        # Normalize and convert to tensor
        transforms.extend([
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])

        return A.Compose(transforms, additional_targets={"image2": "image"})

    def _extract_crop(self, img: np.ndarray, row: int, col: int) -> np.ndarray:
        """Extract a crop from the image at the specified grid position."""
        y_start = row * self.crop_size
        x_start = col * self.crop_size
        return img[y_start:y_start + self.crop_size, x_start:x_start + self.crop_size]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        img_idx, row, col = self.samples[idx]
        name = self._image_files[img_idx]
        stem = osp.splitext(name)[0]

        # Load full images
        before = np.array(Image.open(osp.join(self._data_dir, "A", name)).convert("RGB"))
        after = np.array(Image.open(osp.join(self._data_dir, "B", name)).convert("RGB"))

        # Load mask (try different extensions)
        mask = None
        for ext in [".png", ".jpg", ".tif", ".tiff", ""]:
            mask_path = osp.join(self._data_dir, self._mask_dir, f"{stem}{ext}")
            if not osp.exists(mask_path) and ext == "":
                mask_path = osp.join(self._data_dir, self._mask_dir, name)
            if osp.exists(mask_path):
                mask = np.array(Image.open(mask_path).convert("L"))
                break

        if mask is None:
            raise FileNotFoundError(f"Mask not found for {name}")

        # Extract crops
        before_crop = self._extract_crop(before, row, col)
        after_crop = self._extract_crop(after, row, col)
        mask_crop = self._extract_crop(mask, row, col)

        # Binarize mask (255 -> 1)
        mask_crop = (mask_crop > 127).astype(np.uint8)

        # Apply transforms
        transformed = self.transform(image=before_crop, image2=after_crop, mask=mask_crop)

        return {
            "A": transformed["image"],
            "B": transformed["image2"],
            "mask": transformed["mask"].long(),
            "name": f"{stem}_r{row}_c{col}",
        }

    def compute_statistics(self, num_samples: int = 1000) -> Dict[str, float]:
        """
        Compute patch statistics for dataset analysis.

        Args:
            num_samples: Number of patches to sample (for speed). Use -1 for all.

        Returns:
            Dict with: total_patches, empty_patches, empty_ratio,
                      avg_change_ratio, median_change_ratio
        """
        import random

        indices = list(range(len(self.samples)))
        if num_samples > 0 and num_samples < len(indices):
            indices = random.sample(indices, num_samples)

        change_ratios = []
        empty_count = 0
        patch_size = self.crop_size * self.crop_size

        for idx in indices:
            img_idx, row, col = self.samples[idx]
            name = self._image_files[img_idx]
            stem = osp.splitext(name)[0]

            # Load mask
            mask = None
            for ext in [".png", ".jpg", ".tif", ".tiff", ""]:
                mask_path = osp.join(self._data_dir, self._mask_dir, f"{stem}{ext}")
                if not osp.exists(mask_path) and ext == "":
                    mask_path = osp.join(self._data_dir, self._mask_dir, name)
                if osp.exists(mask_path):
                    mask = np.array(Image.open(mask_path).convert("L"))
                    break

            if mask is None:
                continue

            mask_crop = self._extract_crop(mask, row, col)
            mask_crop = (mask_crop > 127).astype(np.uint8)

            change_pixels = mask_crop.sum()
            change_ratio = change_pixels / patch_size
            change_ratios.append(change_ratio)

            if change_pixels == 0:
                empty_count += 1

        return {
            "total_patches": len(self.samples),
            "sampled_patches": len(indices),
            "empty_patches": empty_count,
            "empty_ratio": empty_count / len(indices) if indices else 0,
            "avg_change_ratio": np.mean(change_ratios) if change_ratios else 0,
            "median_change_ratio": np.median(change_ratios) if change_ratios else 0,
            "std_change_ratio": np.std(change_ratios) if change_ratios else 0,
        }


# =============================================================================
# Dataset Registry
# =============================================================================

DATASETS = {
    "fotbcd": FOTBCDDataset,
    "levircd+": ABLabelDataset,
    "whucd": ABLabelDataset,
}


def get_dataset(
    name: str,
    root: str,
    split: str = "train",
    img_size: int = 256,
    augment: bool = False,
    **kwargs,
) -> Dataset:
    """
    Get dataset by name.

    Args:
        name: Dataset name (fotbcd, levir, whu, sysu, folder)
        root: Dataset root directory
        split: train/val/test
        img_size: Output image size
        augment: Enable augmentations
        **kwargs: Additional dataset-specific arguments

    Returns:
        Dataset instance
    """
    name = name.lower().replace("-", "").replace("_", "")

    if name not in DATASETS and name.replace("cd", "") not in DATASETS:
        raise ValueError(f"Unknown dataset: {name}. Available: {list(DATASETS.keys())}")

    dataset_cls = DATASETS.get(name, DATASETS.get(name.replace("cd", ""), ABLabelDataset))
    return dataset_cls(root=root, split=split, img_size=img_size, augment=augment, **kwargs)


def get_dataloaders(
    name: str,
    root: str,
    batch_size: int = 8,
    img_size: int = 256,
    num_workers: int = 4,
    quick_test: bool = False,
    quick_test_train_samples: int = 100,
    quick_test_val_samples: int = 50,
    **kwargs,
) -> Tuple[DataLoader, DataLoader, Optional[DataLoader]]:
    """
    Create train/val/test dataloaders.

    Args:
        name: Dataset name
        root: Dataset root (contains train/val/test splits)
        batch_size: Batch size
        img_size: Image size
        num_workers: Number of workers

    Returns:
        (train_loader, val_loader, test_loader or None)
    """
    train_dataset = get_dataset(name, root, "train", img_size, augment=True, **kwargs)

    # Try val split first, fallback to test (LEVIR only has train/test)
    try:
        val_dataset = get_dataset(name, root, "val", img_size, augment=False, **kwargs)
    except (FileNotFoundError, ValueError, Exception):
        val_dataset = get_dataset(name, root, "test", img_size, augment=False, **kwargs)
        print("Note: No 'val' split found, using 'test' split for validation")

    # Try to load separate test set
    try:
        test_dataset = get_dataset(name, root, "test", img_size, augment=False, **kwargs)
    except (FileNotFoundError, ValueError, Exception):
        test_dataset = None

    # Apply subset for quick test mode
    if quick_test:
        train_dataset = Subset(train_dataset, range(min(quick_test_train_samples, len(train_dataset))))
        val_dataset = Subset(val_dataset, range(min(quick_test_val_samples, len(val_dataset))))
        if test_dataset:
            test_dataset = Subset(test_dataset, range(min(quick_test_val_samples, len(test_dataset))))
        print(f"Quick test mode: train={len(train_dataset)}, val={len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    ) if test_dataset else None

    return train_loader, val_loader, test_loader


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    print("Available datasets:", list(DATASETS.keys()))


