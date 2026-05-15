"""Central constants for the Smolen-style similarity-learning pipeline.

Anything that is shared across data.py / model.py / train.py and that
might be tuned belongs here. Per-run hyperparameters that change between
experiments stay as CLI args on the training scripts.
"""
from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR     = PROJECT_ROOT / "data"
PARQUET_PATH = DATA_DIR / "processed" / "all_spectra.parquet"
SPLITS_PATH  = DATA_DIR / "processed" / "splits.json"
RUNS_DIR     = PROJECT_ROOT / "runs"

# Class vocabulary. Fixed alphabetical order so that label indices are
# stable across runs and saved checkpoints remain compatible.
POLYMER_CLASSES: tuple[str, ...] = ("HDPE", "LDPE", "PET", "PP", "PS", "PVC")
CLASS_TO_IDX: dict[str, int]     = {c: i for i, c in enumerate(POLYMER_CLASSES)}
IDX_TO_CLASS: dict[int, str]     = {i: c for i, c in enumerate(POLYMER_CLASSES)}
NUM_CLASSES                      = len(POLYMER_CLASSES)

# Spectrum geometry. The 882-channel canonical grid comes from
# compile_data.py; we zero-pad to 896 so that valid-padded convolutions
# in the Smolen architecture land cleanly at the published 52*64=3328
# flatten size.
INPUT_LEN_RAW    = 882
INPUT_LEN_PADDED = 896
PAD_LEFT         = (INPUT_LEN_PADDED - INPUT_LEN_RAW) // 2  # 7
PAD_RIGHT        = INPUT_LEN_PADDED - INPUT_LEN_RAW - PAD_LEFT  # 7

# Smolen Fig. 1A architecture.
EMBED_DIM     = 24
CONV_CHANNELS = (1, 32, 64, 64, 64)   # (in, out1, out2, out3, out4)
CONV_KERNEL   = 5
DENSE_DIMS    = (3328, 160, 120)      # flatten -> FC1 -> FC2 -> (EMBED_DIM or NUM_CLASSES)

# Split protocol (sample-level via StratifiedGroupKFold). Fractions are
# approximate because the splitter rounds to whole groups per fold.
SPLIT_FRACTIONS = {"train": 0.70, "val": 0.10, "calib": 0.10, "test": 0.10}

# Default seed used wherever reproducibility matters. Override per run.
DEFAULT_SEED = 42
