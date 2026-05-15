"""Seed, device, checkpoint, and git utilities shared across train scripts."""
from __future__ import annotations

import json
import random
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch


def set_seeds(seed: int) -> None:
    """Set seeds across random, numpy, and torch (CPU + CUDA).

    cudnn determinism trades a few percent training throughput for bitwise
    reproducibility of the conv kernels. Worth it for a research POC.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def git_commit_hash() -> str | None:
    """Short SHA of the working tree, or None if not in a git checkout."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


@dataclass
class CheckpointPayload:
    epoch: int
    model_state: dict[str, Any]
    optimizer_state: dict[str, Any]
    scheduler_state: dict[str, Any] | None
    metric_value: float
    metric_name: str
    seed: int
    git_sha: str | None = field(default_factory=git_commit_hash)
    extra: dict[str, Any] | None = None


def save_checkpoint(path: Path, payload: CheckpointPayload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(asdict(payload), path)


def load_checkpoint(path: Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    return torch.load(path, map_location=map_location, weights_only=False)


class CSVLogger:
    """Append-only CSV logger that tolerates new columns appearing mid-run."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fieldnames: list[str] | None = None

    def log(self, row: dict[str, Any]) -> None:
        import csv
        if self._fieldnames is None:
            self._fieldnames = list(row.keys())
            with self.path.open("w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=self._fieldnames)
                w.writeheader()
                w.writerow(row)
        else:
            with self.path.open("a", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=self._fieldnames, extrasaction="ignore")
                w.writerow(row)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        json.dump(obj, fh, indent=2, sort_keys=True)


def read_json(path: Path) -> Any:
    with path.open() as fh:
        return json.load(fh)
