"""Parquet -> Dataset -> DataLoader pipeline for the Smolen-style model.

Responsibilities:
  - load the harmonized parquet produced by compile_data.py
  - derive a stable physical_sample_id (identity for the current dataset;
    placeholder for future replicate groups)
  - peel the data into train / val / calib / test at the *sample* level
    via StratifiedGroupKFold so that no physical sample appears on both
    sides of a split
  - per-spectrum min/max normalize and zero-pad to 896 channels once at
    Dataset construction
  - build a DataLoader with pytorch-metric-learning's MPerClassSampler
    for the embedder training loop; a plain shuffled loader for the
    classification-head baseline; and a plain unshuffled loader for
    val / calib / test
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from pytorch_metric_learning.samplers import MPerClassSampler
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader, Dataset

from src.config import (
    CLASS_TO_IDX,
    INPUT_LEN_RAW,
    PAD_LEFT,
    PAD_RIGHT,
    PARQUET_PATH,
    POLYMER_CLASSES,
    SPLITS_PATH,
)
from src.utils import read_json, write_json


# ---------------------------------------------------------------------------
# Parquet loading and physical-sample derivation
# ---------------------------------------------------------------------------


def load_parquet(path: Path | str = PARQUET_PATH) -> pd.DataFrame:
    """Load the harmonized parquet, filter to the 6 target classes, and
    attach a `physical_sample_id` column used for group-aware splitting.
    """
    df = pd.read_parquet(path)
    df = df[df["polymer_class_raw"].isin(POLYMER_CLASSES)].reset_index(drop=True)
    df["physical_sample_id"] = derive_physical_sample_id(df)
    return df


_REPLICATE_SUFFIX = re.compile(r"[._\-\s](?:rep|r|run|m|meas)?[._\-]?\d{1,3}$", re.IGNORECASE)


def derive_physical_sample_id(df: pd.DataFrame) -> pd.Series:
    """Map each `sample_id` to a physical-sample identifier.

    For the current FLOPP / FLOPP-e / Villegas data each row is its own
    physical particle, so identity is correct. Trailing numeric replicate
    suffixes (e.g. `..._001`, `... rep2`) are stripped defensively so the
    same function keeps working if multi-shot data is added later.
    """
    base = df["sample_id"].astype(str).str.replace(_REPLICATE_SUFFIX, "", regex=True)
    return df["source"].astype(str) + "::" + base


# ---------------------------------------------------------------------------
# Sample-level splitting via StratifiedGroupKFold
# ---------------------------------------------------------------------------


def make_splits(df: pd.DataFrame, seed: int) -> dict[str, str]:
    """Four-way sample-level split: train / val / calib / test.

    Implemented as three sequential `StratifiedGroupKFold` peels (10, 9, 8
    splits) so that each peel removes ~10% of the *total* dataset while
    keeping every physical sample on exactly one side of every boundary.
    """
    y = df["polymer_class_raw"].to_numpy()
    g = df["physical_sample_id"].to_numpy()
    idx_all = np.arange(len(df))

    test_idx = _peel(idx_all, y, g, n_splits=10, seed=seed)
    remaining = np.setdiff1d(idx_all, test_idx, assume_unique=False)

    calib_idx = _peel(remaining, y[remaining], g[remaining], n_splits=9, seed=seed + 1)
    remaining = np.setdiff1d(remaining, calib_idx, assume_unique=False)

    val_idx = _peel(remaining, y[remaining], g[remaining], n_splits=8, seed=seed + 2)
    train_idx = np.setdiff1d(remaining, val_idx, assume_unique=False)

    assignment: dict[str, str] = {}
    for sid, split in [
        (df["spectrum_id"].iloc[train_idx], "train"),
        (df["spectrum_id"].iloc[val_idx],   "val"),
        (df["spectrum_id"].iloc[calib_idx], "calib"),
        (df["spectrum_id"].iloc[test_idx],  "test"),
    ]:
        for s in sid:
            assignment[s] = split

    _assert_no_group_leakage(df, assignment)
    return assignment


def _peel(idx: np.ndarray, y: np.ndarray, g: np.ndarray, n_splits: int, seed: int) -> np.ndarray:
    """Run one StratifiedGroupKFold round and return the first fold's
    held-out indices, translated back into the original index space."""
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    _, local_held = next(sgkf.split(X=np.zeros(len(idx)), y=y, groups=g))
    return idx[local_held]


def _assert_no_group_leakage(df: pd.DataFrame, assignment: dict[str, str]) -> None:
    by_split: dict[str, set[str]] = {"train": set(), "val": set(), "calib": set(), "test": set()}
    for _, row in df.iterrows():
        by_split[assignment[row["spectrum_id"]]].add(row["physical_sample_id"])
    splits = list(by_split)
    for i, a in enumerate(splits):
        for b in splits[i + 1:]:
            overlap = by_split[a] & by_split[b]
            if overlap:
                raise AssertionError(f"physical_sample_id leakage between {a!r} and {b!r}: {sorted(overlap)[:5]}")


def save_splits(assignment: dict[str, str], path: Path | str = SPLITS_PATH) -> None:
    write_json(Path(path), assignment)


def load_splits(path: Path | str = SPLITS_PATH) -> dict[str, str]:
    return read_json(Path(path))


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class SpectrumDataset(Dataset):
    """In-memory dataset of preprocessed FTIR spectra.

    The full corpus is ~3k spectra of 882 floats each (~11 MB), so we
    pre-normalize and pre-pad once at construction. `__getitem__` is then
    a pure tensor slice.
    """

    def __init__(self, df: pd.DataFrame, split_ids: Iterable[str]) -> None:
        split_id_set = set(split_ids)
        view = df[df["spectrum_id"].isin(split_id_set)].reset_index(drop=True)
        if view.empty:
            raise ValueError("SpectrumDataset received an empty split")

        intensities = np.stack(view["intensity"].to_numpy()).astype(np.float32)
        if intensities.shape[1] != INPUT_LEN_RAW:
            raise ValueError(
                f"Expected {INPUT_LEN_RAW}-channel spectra, got {intensities.shape[1]}"
            )

        # Per-spectrum min/max normalize to [0, 1].
        lo = intensities.min(axis=1, keepdims=True)
        hi = intensities.max(axis=1, keepdims=True)
        intensities = (intensities - lo) / (hi - lo + 1e-8)

        # Zero-pad 882 -> 896 so the valid-padded conv stack ends at the
        # Smolen-published 52*64 flatten size.
        intensities = np.pad(
            intensities,
            ((0, 0), (PAD_LEFT, PAD_RIGHT)),
            mode="constant",
            constant_values=0.0,
        )

        # Add channel dim for Conv1d: (N, 1, 896).
        self.X = torch.from_numpy(intensities[:, None, :]).contiguous()
        self.y = torch.tensor(
            [CLASS_TO_IDX[c] for c in view["polymer_class_raw"]], dtype=torch.long
        )
        self.spectrum_ids: list[str] = view["spectrum_id"].tolist()

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[i], self.y[i]


# ---------------------------------------------------------------------------
# DataLoaders
# ---------------------------------------------------------------------------


def build_train_loader_pairmining(
    dataset: SpectrumDataset,
    batch_size: int = 48,
    m: int = 8,
    num_workers: int = 0,
) -> DataLoader:
    """DataLoader for pair/triplet metric learning.

    `MPerClassSampler` guarantees `m` examples per sampled class so that
    every batch contains the positives and negatives that the miner
    needs. The library imposes `m * num_unique_classes >= batch_size`;
    the defaults above satisfy that with one class-balanced batch per
    "pass".
    """
    n_classes = len(set(dataset.y.tolist()))
    if m * n_classes < batch_size:
        raise ValueError(
            f"MPerClassSampler requires m * num_classes >= batch_size; "
            f"got m={m}, num_classes={n_classes}, batch_size={batch_size}."
        )
    sampler = MPerClassSampler(
        labels=dataset.y.tolist(),
        m=m,
        batch_size=batch_size,
        length_before_new_iter=len(dataset),
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        drop_last=True,
        pin_memory=torch.cuda.is_available(),
    )


def build_shuffled_loader(
    dataset: SpectrumDataset,
    batch_size: int = 64,
    num_workers: int = 0,
) -> DataLoader:
    """DataLoader for the cross-entropy classification baseline."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=False,
        pin_memory=torch.cuda.is_available(),
    )


def build_eval_loader(
    dataset: SpectrumDataset,
    batch_size: int = 128,
    num_workers: int = 0,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
        pin_memory=torch.cuda.is_available(),
    )


# ---------------------------------------------------------------------------
# Convenience: split-prep entry point
# ---------------------------------------------------------------------------


def prepare_splits(seed: int, force: bool = False) -> dict[str, str]:
    """Compute (or reload) the persistent train/val/calib/test assignment.

    The assignment is persisted to disk so that repeated runs draw from
    the same partition; pass `force=True` to recompute.
    """
    if SPLITS_PATH.exists() and not force:
        return load_splits()
    df = load_parquet()
    assignment = make_splits(df, seed=seed)
    save_splits(assignment)
    return assignment
