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
SPLITS_DIR   = DATA_DIR / "processed"
RUNS_DIR     = PROJECT_ROOT / "runs"


def splits_path(mode: str) -> Path:
    """Path to the persisted split assignment for a given splitting mode."""
    return SPLITS_DIR / f"splits_{mode}.json"


# Source-out split: which sources go into the train pool vs. into the
# held-out test set. The point is to evaluate on data the model never
# saw at the *source* level, so cross-instrument generalization is
# measurable rather than masked by random in-source splits.
SOURCE_OUT_TRAIN: tuple[str, ...] = ("Villegas-c4", "FLOPP")
SOURCE_OUT_TEST:  tuple[str, ...] = ("FLOPP-e", "OpenSpecy")

# source_cv split: a cross-source validation signal without saturating on
# Villegas. Construction:
#   - val: a BALANCED per-class quota of physical samples drawn from the
#     non-Villegas pool, taking each class from whichever source has the
#     most of it (minimizes test damage), capped so no single source loses
#     more than VAL_MAX_SHARE of its samples of a class (avoids draining a
#     thin source). This keeps OpenSpecy — our only 6-class cross-lab test —
#     almost entirely intact.
#   - train: Villegas + FLOPP (backbone) + the in-house non-val remainder's
#     train portion (injects deployment-instrument structure), minus a calib
#     carve from the backbone.
#   - test: in-house non-val remainder's test portion (test_In-house) +
#     FLOPP-e and OpenSpecy non-val remainders (test_<source>).
# Caveat: any source a class is drawn into val from has a mildly optimistic
# (selection-biased) test number for that class; the quota is kept small to
# bound this.
SOURCE_CV_TRAIN: tuple[str, ...]      = ("Villegas-c4", "FLOPP")   # backbone -> train (+ calib)
SOURCE_CV_VAL_SOURCES: tuple[str, ...] = ("In-house", "FLOPP-e", "OpenSpecy")  # val drawn from these
SOURCE_CV_VAL_QUOTA: int               = 5      # target physical samples per class in val
SOURCE_CV_VAL_MAX_SHARE: float         = 0.5    # never take >this share of a source's samples of a class
SOURCE_CV_INHOUSE_TEST_FRAC: float     = 0.34   # in-house non-val remainder: this -> test, rest -> train
SOURCE_CV_TEST: tuple[str, ...]        = ("FLOPP-e", "OpenSpecy")  # non-val remainder -> locked test

# source_cv2 split: the multi-instrument, full-corpus protocol. Trains on
# FIVE instruments so the encoder can learn instrument-invariance rather than
# one instrument's fingerprint, and tests on whole held-out instruments for an
# honest hardware-agnostic number with a large (tight-CI) pooled test set.
#   - train: Villegas + Cowger + Poseidon + FLOPP (backbone) + the in-house
#     non-val/non-test remainder. Cowger uniquely adds explicit HDPE/LDPE on a
#     second instrument — the best lever for the hardest (PE-vs-PE) error.
#   - val: a BALANCED per-class quota of physical samples drawn ROUND-ROBIN
#     across the held-in sources (so the selection signal is multi-instrument,
#     not just the tiny in-house set), capped per source so none is drained.
#   - test (locked, whole-instrument, never trained): OpenSpecy (multi-lab),
#     FLOPP-e (weathered), Baskaran (food-packaging multilayer), plus the
#     in-house test slice (deployment instrument).
# Note: Poseidon contributes a single noisy marine PVC spectrum to train; it
# is harmless and left in. PVC remains a near-Villegas monoculture in training
# (almost no non-Villegas PVC exists), so PVC cross-hardware stays the weakest.
SOURCE_CV2_TRAIN: tuple[str, ...]       = ("Villegas-c4", "Cowger", "Poseidon", "FLOPP")  # -> train (+calib)
SOURCE_CV2_MIXED: str                  = "In-house"   # split by sample: train + locked test slice
SOURCE_CV2_TEST: tuple[str, ...]       = ("OpenSpecy", "FLOPP-e", "Baskaran")  # locked whole-instrument test
SOURCE_CV2_VAL_SOURCES: tuple[str, ...] = ("Villegas-c4", "Cowger", "Poseidon", "FLOPP", "In-house")  # held-in; val drawn round-robin from these
SOURCE_CV2_VAL_QUOTA: int              = 8      # target physical samples per class in val
SOURCE_CV2_VAL_MAX_SHARE: float        = 0.5    # never take >this share of a source's samples of a class
SOURCE_CV2_INHOUSE_TEST_FRAC: float    = 0.34   # in-house non-val remainder: this -> test, rest -> train

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

# Endpoints of the canonical wavenumber grid (cm-1). compile_data.py
# resamples every training spectrum onto np.linspace(LO, HI, INPUT_LEN_RAW);
# inference must use the identical grid.
CANONICAL_LO = 700.0
CANONICAL_HI = 3996.0

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

# On-the-fly physics-based augmentation (training split only). See
# src/augment.py and research-notes/augmentation_implementation_report.md.
# Every transform is wavenumber-dependent or per-channel: per-spectrum
# min/max normalization downstream cancels pure offsets and global scales,
# so those are deliberately absent. Axis transforms (wn_shift, broaden) are
# kept mild to avoid erasing the narrow HDPE<->LDPE discriminators.
AUG = {
    "enabled": True,
    "p_each": 0.7,                                                  # per-transform fire probability
    "mult_field":  {"c1": (-0.6, 0.6), "c2": (-0.3, 0.3), "clip": (0.5, 1.8)},  # 4.1 penetration tilt + scatter
    "add_baseline": {"beta": (0.01, 0.08), "order": 2},            # 4.2 sloped/curved baseline (1-8% of range)
    "noise":       {"rho": (0.005, 0.02)},                         # 4.3 Gaussian noise (0.5-2% of max)
    "wn_shift":    {"channels": (-1.0, 1.0)},                      # 4.4 PE-risky; widen to (-2,2) only if PE holds
    "broaden":     {"sigma_ch": (0.0, 1.0)},                       # 4.5 PE-risky; keep <=1 channel
    "curriculum_warmup_epochs": 0,                                 # set ~10 if early epochs unstable
}
