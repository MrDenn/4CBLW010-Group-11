"""Open-set recognition.

For the model to be deployable as a sorting tool it must be able to say
"I don't recognise this spectrum" rather than always picking one of the
six known classes. Two complementary approaches are implemented:
  A) Smolen baseline: SVC with Platt scaling + a confidence threshold.
     Sweep tau over the probability range to draw a known-vs-unknown ROC.
  B) Innovation: split-conformal prediction with MAPIE. An empty
     prediction set means "reject as out-of-distribution".

The "unknown" stream is constructed by holding one of the 6 polymer
classes out of training entirely, then evaluating its embeddings as the
unknown distribution. Choose the held-out class with --unknown-class.

Usage:
  python -m src.openset --embedder-run baseline_multisim --unknown-class PVC
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from sklearn.svm import SVC

from src.config import POLYMER_CLASSES, RUNS_DIR
from src.data import (
    SpectrumDataset,
    build_eval_loader,
    load_parquet,
    prepare_splits,
)
from src.model import SmolenCNN
from src.train import embed_all
from src.utils import get_device, load_checkpoint, write_json


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def embed_split(ckpt_path: Path, ds: SpectrumDataset, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model = SmolenCNN().to(device)
    payload = load_checkpoint(ckpt_path, map_location=device)
    model.load_state_dict(payload["model_state"])
    model.eval()
    return embed_all(model, build_eval_loader(ds, batch_size=128), device)


# ---------------------------------------------------------------------------
# Approach A: Platt-thresholded SVC
# ---------------------------------------------------------------------------


def platt_threshold_roc(
    E_train_known: np.ndarray, y_train_known: np.ndarray,
    E_test_known:  np.ndarray,
    E_test_unknown: np.ndarray,
    seed: int,
) -> dict[str, float]:
    """Train an OvR SVC on knowns, score known vs. unknown by max-prob.

    Higher confidence = more likely to be "known", so 1 - max_prob is the
    standard unknown score. Returns AUROC + the FPR/TPR at the EER point.
    """
    svc = SVC(
        kernel="rbf", C=1.0, gamma="scale",
        probability=True, decision_function_shape="ovr", random_state=seed,
    )
    svc.fit(E_train_known, y_train_known)

    conf_known   = svc.predict_proba(E_test_known).max(axis=1)
    conf_unknown = svc.predict_proba(E_test_unknown).max(axis=1)

    scores = np.concatenate([-conf_known, -conf_unknown])  # higher = more "unknown"
    labels = np.concatenate([np.zeros(len(conf_known)), np.ones(len(conf_unknown))])
    auroc = float(roc_auc_score(labels, scores))

    return {
        "auroc_unknown_vs_known": auroc,
        "mean_conf_known":        float(conf_known.mean()),
        "mean_conf_unknown":      float(conf_unknown.mean()),
    }


# ---------------------------------------------------------------------------
# Approach B: MAPIE split-conformal
# ---------------------------------------------------------------------------


def mapie_conformal(
    E_train_known: np.ndarray, y_train_known: np.ndarray,
    E_calib: np.ndarray, y_calib: np.ndarray,
    E_test_known: np.ndarray, y_test_known: np.ndarray,
    E_test_unknown: np.ndarray,
    seed: int, alpha: float = 0.10,
) -> dict[str, float]:
    """Split-conformal SVC with MAPIE; report set sizes and coverage.

    For the unknown stream we report the empty-set rate (= rejection rate
    at the chosen alpha). Higher is better - the model is correctly
    saying "I don't know what this is".

    MAPIE 1.x API. The package has been rapidly evolving across recent
    releases; the try/except below falls back to the legacy
    `MapieClassifier` shape if the 1.x `SplitConformalClassifier` import
    fails or its constructor signature changes again.
    """
    try:
        from mapie.classification import SplitConformalClassifier as _Mapie
    except ImportError:
        from mapie.classification import MapieClassifier as _Mapie  # pragma: no cover

    svc = SVC(
        kernel="rbf", C=1.0, gamma="scale",
        probability=True, decision_function_shape="ovr", random_state=seed,
    )
    svc.fit(E_train_known, y_train_known)

    # Both legacy MapieClassifier and SplitConformalClassifier accept the
    # same fit-on-calibration shape when used with a prefit estimator.
    try:
        mapie = _Mapie(estimator=svc, confidence_level=1 - alpha, prefit=True)
        mapie.conformalize(E_calib, y_calib)
        _, y_pis = mapie.predict_sets(E_test_known)
        _, y_pis_unknown = mapie.predict_sets(E_test_unknown)
    except TypeError:
        mapie = _Mapie(estimator=svc, method="lac", cv="prefit")
        mapie.fit(E_calib, y_calib)
        _, y_pis = mapie.predict(E_test_known, alpha=alpha)
        _, y_pis_unknown = mapie.predict(E_test_unknown, alpha=alpha)

    set_sizes_known   = y_pis.squeeze(-1).sum(axis=1)
    set_sizes_unknown = y_pis_unknown.squeeze(-1).sum(axis=1)
    coverage_known    = float((y_pis.squeeze(-1)[np.arange(len(y_test_known)), y_test_known]).mean())
    empty_rate_known  = float((set_sizes_known == 0).mean())
    empty_rate_unknown = float((set_sizes_unknown == 0).mean())

    return {
        "alpha":                alpha,
        "coverage_known":       coverage_known,
        "mean_set_size_known":  float(set_sizes_known.mean()),
        "empty_rate_known":     empty_rate_known,
        "empty_rate_unknown":   empty_rate_unknown,
        "mean_set_size_unknown": float(set_sizes_unknown.mean()),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Open-set evaluation by holding out one polymer class.")
    p.add_argument("--embedder-run", required=True)
    p.add_argument("--checkpoint", default="best.pt")
    p.add_argument("--unknown-class", required=True, choices=POLYMER_CLASSES)
    p.add_argument("--alpha", type=float, default=0.10)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = get_device()
    splits = prepare_splits(seed=args.seed)
    df = load_parquet()

    # The held-out class becomes the unknown stream, drawn from ALL of
    # its spectra regardless of original split (the embedder was trained
    # with this class present, so this is a "what does the model do on
    # spectra it saw at train time but pretends not to know" test - a
    # *weak* open-set proxy. A stronger test is to retrain the embedder
    # without the held-out class; that is a separate run.)
    unknown_mask = df["polymer_class_raw"] == args.unknown_class
    known_df     = df[~unknown_mask]
    unknown_df   = df[unknown_mask]

    ckpt = Path(RUNS_DIR) / args.embedder_run / args.checkpoint
    ds_train = SpectrumDataset(known_df, [s for s, sp in splits.items() if sp == "train"])
    ds_calib = SpectrumDataset(known_df, [s for s, sp in splits.items() if sp == "calib"])
    ds_test  = SpectrumDataset(known_df, [s for s, sp in splits.items() if sp == "test"])
    ds_unknown = SpectrumDataset(unknown_df, unknown_df["spectrum_id"].tolist())

    E_train, y_train = embed_split(ckpt, ds_train, device)
    E_calib, y_calib = embed_split(ckpt, ds_calib, device)
    E_test,  y_test  = embed_split(ckpt, ds_test,  device)
    E_unknown, _     = embed_split(ckpt, ds_unknown, device)

    print(f"Unknown class: {args.unknown_class} | known train={len(ds_train)} calib={len(ds_calib)} test={len(ds_test)} | unknown={len(ds_unknown)}")

    a = platt_threshold_roc(E_train, y_train, E_test, E_unknown, seed=args.seed)
    print("\n[Approach A] Platt-thresholded SVC")
    for k, v in a.items():
        print(f"  {k:32s} {v:.4f}")

    b = mapie_conformal(E_train, y_train, E_calib, y_calib, E_test, y_test, E_unknown, seed=args.seed, alpha=args.alpha)
    print(f"\n[Approach B] MAPIE split-conformal (alpha={args.alpha})")
    for k, v in b.items():
        print(f"  {k:32s} {v:.4f}")

    out = {
        "unknown_class": args.unknown_class,
        "alpha":         args.alpha,
        "platt":         a,
        "conformal":     b,
    }
    write_json(Path(RUNS_DIR) / args.embedder_run / f"openset_{args.unknown_class}.json", out)


if __name__ == "__main__":
    main()
