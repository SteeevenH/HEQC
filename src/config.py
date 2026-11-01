from __future__ import annotations

from pathlib import Path

# Project Paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
WSI_DIR = PROJECT_ROOT / "wsi"
MODEL_DIR = PROJECT_ROOT / "model"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

# Model
MODEL_NAME = "efficientnet_b0"  # selects both model/<name>.pt and model/threshold_<name>.json.
THRESHOLD_PROFILE = "balanced"  # balanced | recall

# WSI processing parameters
TARGET_MAGNIFICATION = 10.0
STRIDE = 256
RECOMMENDED_MPP_MIN = 0.85
RECOMMENDED_MPP_MAX = 1.15

# Use manual magnification only when scanner metadata is missing.
# Otherwise, rely on the magnification reported by the WSI.
MANUAL_OBJECTIVE_POWER = None
MANUAL_BASE_MPP = None

# Hardware compatibility (one device only; no DataParallel)
DEVICE = "auto"  # auto | cuda | mps | cpu
CUDA_DEVICE_INDEX = 0
BATCH_SIZE = 64
NUM_WORKERS = 8
PREFETCH_FACTOR = 2
PIN_MEMORY = True
PERSISTENT_WORKERS = True
USE_AMP = False

# Visualization parameters
HEATMAP_MAGNIFICATION = 1.25
THUMBNAIL_MAGNIFICATION = 1.25
READ_TILE_SIZE = 2048

HEATMAP_MAX_ALPHA = 0.65
HEATMAP_ALPHA_POWER = 1.5
HEATMAP_BASE_SATURATION = 0.70
HEATMAP_BASE_BRIGHTNESS = 0.92

# None: automatic radius based on projected patch overlap.
# A numeric value sets the blur radius in heatmap pixels.
SMOOTHING_RADIUS_OVERRIDE = None

PNG_COMPRESSION = 4

# Runtime behavior
OVERWRITE = False
CONTINUE_ON_ERROR = True
SAVE_PATCH_PREDICTIONS = False
RANDOM_SEED = 42


def model_path() -> Path:
    return MODEL_DIR / f"{MODEL_NAME}.pt"
def threshold_path() -> Path:
    return MODEL_DIR / f"threshold_{MODEL_NAME}.json"
