"""
Configuration for Change Detection Benchmark.

Usage:
    # Set dataset via environment variable or edit DATASET below
    DATASET=levircd+ python train.py
    DATASET=fotbcd python train.py
    DATASET=whucd python train.py
"""

import os
import torch


# Dataset presets - dataset-specific paths and settings
# All datasets use 256x256 patches for fair comparison
DATASET_PRESETS = {
    "fotbcd": {
        "data_root": "/workspace/datasets/FOTBCD-Binary",
        "img_size": 256,
        "crop_size": 256,
        "original_size": 512,
        "batch_size": 128,
    },
    "levircd+": {
        "data_root": "/workspace/datasets/LEVIR-CD+",
        "img_size": 256,
        "crop_size": 256,
        "original_size": 1024,
        "batch_size": 128,
    },
    "whucd": {
        "data_root": "/workspace/datasets/WHU-CD",
        "img_size": 256,
        "crop_size": 256,
        "original_size": 256,
        "batch_size": 128,
    },
}


class CFG:
    # ==========================================================================
    # Dataset selection (override via env: DATASET=fotbcd python train.py)
    # ==========================================================================
    DATASET = os.environ.get("DATASET", "fotbcd")

    # Load dataset-specific settings
    _preset = DATASET_PRESETS.get(DATASET, DATASET_PRESETS["fotbcd"])
    DATA_ROOT = _preset["data_root"]
    IMG_SIZE = _preset["img_size"]
    BATCH_SIZE = _preset["batch_size"]
    CROP_SIZE = _preset.get("crop_size", 256)
    ORIGINAL_SIZE = _preset.get("original_size", 256)

    # ==========================================================================
    # Device
    # ==========================================================================
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ==========================================================================
    # Model (shared across all datasets)
    # ==========================================================================
    MODEL_NAME = "vit_large_patch16_dinov3.sat493m"
    ENCODER = MODEL_NAME
    PRETRAINED = True
    ENCODER_DIM = 256
    PATCH_SIZE = 16
    IN_CHANNELS = 3
    NUM_CLASSES = 2
    FREEZE_ENCODER = True
    USE_REFINEMENT = True
    SHALLOW_DIM = 32

    # ==========================================================================
    # Training (step-based for fair cross-dataset comparison)
    # ==========================================================================
    TOTAL_STEPS = 50000  # Same for all datasets (fair comparison)
    LR = 4e-4
    WEIGHT_DECAY = 1e-4
    OPTIMIZER = "adamw"
    LR_SCHEDULER = "cosine"
    WARMUP_STEPS = 2000  # Step-based warmup

    # ==========================================================================
    # Loss
    # ==========================================================================
    LOSS_TYPE = "sharp"
    BCE_WEIGHT = 1.0
    DICE_WEIGHT = 1.0
    FOCAL_GAMMA = 2.0
    LOVASZ_WEIGHT = 1.0
    BOUNDARY_WEIGHT = 1.5
    EDGE_MINING_WEIGHT = 3.0

    # ==========================================================================
    # Data loading
    # ==========================================================================
    NUM_WORKERS = 4
    PIN_MEMORY = True
    USE_AMP = True

    # ==========================================================================
    # Step-based logging (for fair cross-dataset comparison)
    # ==========================================================================
    LOG_EVERY_STEPS = 500  # Log train metrics every N steps
    VAL_EVERY_STEPS = 1000  # Run validation every N steps
    SAVE_EVERY_STEPS = 5000  # Save checkpoint every N steps
    CROSS_EVAL_EVERY_STEPS = 2000  # Cross-dataset evaluation every N steps

    # ==========================================================================
    # Visualization
    # ==========================================================================
    N_VIS_SAMPLES = 32
    SAVE_DIR = "./runs"
    # ==========================================================================
    # Experiment naming
    # ==========================================================================
    EXPERIMENT_NAME = f"{DATASET}_{IMG_SIZE}px_bs{BATCH_SIZE}"

    # ==========================================================================
    # Quick test mode
    # ==========================================================================
    QUICK_TEST = False
    QUICK_TEST_TRAIN_SAMPLES = 100
    QUICK_TEST_VAL_SAMPLES = 50
