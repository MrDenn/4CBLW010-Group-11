"""Closed-set calibrated probabilities (stage 1 of the confidence head).

The embedder + k-NN tells us *which* of the six plastics a spectrum is,
but not *how sure* in any trustworthy sense - k-NN vote fractions are
coarse (only k+1 distinct values) and uncalibrated. This module attaches
an honest probability P(class | known) to each prediction, staying inside
the same distance-on-embeddings paradigm the rest of the pipeline uses:

  1. Build one prototype per class = the mean training embedding of that
     class (gallery centroid).
  2. Score a spectrum by squared-Euclidean distance to each prototype;
     logits = -dist^2.  This is a Gaussian-prototype classifier.
  3. Calibrate a single temperature T on the held-out `calib` split by
     minimizing NLL (Guo et al. 2017 temperature scaling). Probabilities
     are softmax(-dist^2 / T).

Temperature scaling is monotone, so it never changes the argmax - the
class decision is unchanged, only the confidence is made honest. We then
report Expected Calibration Error / NLL / Brier before (T=1) vs after
(T*), the reliability diagram, and agreement with the established k-NN
classifier, so the calibrated head is shown to reproduce the headline
accuracy while fixing the confidence.

This is stage 1 of the two-stage confidence design (see
research-notes/development_decision_log.md section 11). The fitted
prototypes + temperature are saved for reuse by the open-set gate and the
fusion layer.

Usage:
  python -m src.closedset --embedder-run cv2_aug \
      --split-mode source_cv2 --checkpoint latest.pt
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: render straight to PNG files

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

from src.config import IDX_TO_CLASS, NUM_CLASSES, POLYMER_CLASSES, RUNS_DIR
from src.data import SpectrumDataset, build_eval_loader, load_parquet, prepare_splits
from src.predict import load_embedder
from src.train import embed_all
from src.utils import get_device, write_json


# ---------------------------------------------------------------------------
# Core: prototypes, distance logits, temperature-scaled softmax
# ---------------------------------------------------------------------------


def class_prototypes(E_train: np.ndarray, y_train: np.ndarray) -> np.ndarray:
    """One prototype per class = mean training embedding (gallery centroid).

    Returns (NUM_CLASSES, D). Asserts every class is present so the row
    index lines up with CLASS_TO_IDX.
    """
    protos = np.zeros((NUM_CLASSES, E_train.shape[1]), dtype=np.float64)
    for c in range(NUM_CLASSES):
        mask = y_train == c
        if not mask.any():
            raise ValueError(f"class {IDX_TO_CLASS[c]} absent from gallery; cannot build prototype")
        protos[c] = E_train[mask].mean(axis=0)
    return protos


def squared_distances(E: np.ndarray, protos: np.ndarray) -> np.ndarray:
    """(N, NUM_CLASSES) squared-Euclidean distance to each prototype."""
    # ||x - mu||^2 = ||x||^2 - 2 x.mu + ||mu||^2
    x2 = (E ** 2).sum(axis=1, keepdims=True)
    m2 = (protos ** 2).sum(axis=1)[None, :]
    return x2 - 2.0 * E @ protos.T + m2


def softmax_from_dist(d2: np.ndarray, T: float) -> np.ndarray:
    """Temperature-scaled softmax over logits = -d2 / T (row-stable)."""
    logits = -d2 / T
    logits = logits - logits.max(axis=1, keepdims=True)
    ex = np.exp(logits)
    return ex / ex.sum(axis=1, keepdims=True)


def _nll(probs: np.ndarray, y: np.ndarray) -> float:
    """Mean negative log-likelihood of the true class."""
    p_true = probs[np.arange(len(y)), y]
    return float(-np.log(np.clip(p_true, 1e-12, None)).mean())


def fit_temperature(d2_calib: np.ndarray, y_calib: np.ndarray) -> float:
    """1-D search for the temperature minimizing NLL on the calib split."""
    obj = lambda logT: _nll(softmax_from_dist(d2_calib, float(np.exp(logT))), y_calib)
    # Optimize in log-space so T stays positive; bounds ~ [e^-4, e^4].
    res = minimize_scalar(obj, bounds=(-4.0, 4.0), method="bounded")
    return float(np.exp(res.x))


# ---------------------------------------------------------------------------
# Calibration diagnostics
# ---------------------------------------------------------------------------


def brier_score(probs: np.ndarray, y: np.ndarray) -> float:
    """Multiclass Brier score (mean squared error vs one-hot)."""
    onehot = np.zeros_like(probs)
    onehot[np.arange(len(y)), y] = 1.0
    return float(((probs - onehot) ** 2).sum(axis=1).mean())


def expected_calibration_error(
    probs: np.ndarray, y: np.ndarray, n_bins: int = 10
) -> tuple[float, list[dict]]:
    """Top-label ECE with equal-width confidence bins.

    Returns (ece, bins) where each bin carries its mean confidence, mean
    accuracy and count - enough to draw a reliability diagram.
    """
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == y).astype(np.float64)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    bins: list[dict] = []
    n = len(y)
    for lo, hi in zip(edges[:-1], edges[1:]):
        # Last bin is closed on the right so conf == 1.0 is counted.
        in_bin = (conf > lo) & (conf <= hi) if hi < 1.0 else (conf > lo) & (conf <= hi + 1e-9)
        cnt = int(in_bin.sum())
        if cnt == 0:
            bins.append({"lo": float(lo), "hi": float(hi), "count": 0,
                         "confidence": float("nan"), "accuracy": float("nan")})
            continue
        bin_conf = float(conf[in_bin].mean())
        bin_acc = float(correct[in_bin].mean())
        ece += (cnt / n) * abs(bin_acc - bin_conf)
        bins.append({"lo": float(lo), "hi": float(hi), "count": cnt,
                     "confidence": bin_conf, "accuracy": bin_acc})
    return float(ece), bins


def _metrics(probs: np.ndarray, y: np.ndarray, n_bins: int) -> dict:
    ece, bins = expected_calibration_error(probs, y, n_bins)
    return {
        "accuracy": float(accuracy_score(y, probs.argmax(axis=1))),
        "nll":      _nll(probs, y),
        "brier":    brier_score(probs, y),
        "ece":      ece,
        "bins":     bins,
    }


# ---------------------------------------------------------------------------
# Reliability diagram
# ---------------------------------------------------------------------------


def save_reliability_diagram(
    before: dict, after: dict, temperature: float, out: Path
) -> None:
    """Two reliability panels (pre/post temperature scaling) side by side."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, m, title in (
        (axes[0], before, f"Uncalibrated (T=1)\nECE={before['ece']:.3f}  NLL={before['nll']:.3f}"),
        (axes[1], after, f"Temperature-scaled (T={temperature:.2f})\nECE={after['ece']:.3f}  NLL={after['nll']:.3f}"),
    ):
        ax.plot([0, 1], [0, 1], "--", color="gray", lw=1, label="perfect")
        xs = [b["confidence"] for b in m["bins"] if b["count"] > 0]
        ys = [b["accuracy"] for b in m["bins"] if b["count"] > 0]
        ws = [b["count"] for b in m["bins"] if b["count"] > 0]
        ax.scatter(xs, ys, s=[20 + 4 * w for w in ws], color="#1565c0", zorder=3)
        ax.plot(xs, ys, color="#1565c0", lw=1, alpha=0.6)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_xlabel("predicted confidence")
        ax.set_ylabel("empirical accuracy")
        ax.set_title(title)
        ax.legend(loc="upper left")
    fig.suptitle("Closed-set probability calibration (pooled held-out test)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out, dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Closed-set calibrated probabilities + diagnostics.")
    p.add_argument("--embedder-run", required=True)
    p.add_argument("--split-mode", choices=("random", "source_out", "source_cv", "source_cv2"),
                   default="source_cv2")
    p.add_argument("--checkpoint", default="latest.pt")
    p.add_argument("--k", type=int, default=5, help="k for the k-NN agreement check.")
    p.add_argument("--bins", type=int, default=10, help="ECE / reliability bins.")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = get_device()
    model = load_embedder(args.embedder_run, args.checkpoint, device)

    splits = prepare_splits(seed=args.seed, mode=args.split_mode)
    df = load_parquet()

    def _embed(split_pred) -> tuple[np.ndarray, np.ndarray]:
        ids = [s for s, sp in splits.items() if split_pred(sp)]
        return embed_all(model, build_eval_loader(SpectrumDataset(df, ids), 128), device)

    E_train, y_train = _embed(lambda sp: sp == "train")
    E_calib, y_calib = _embed(lambda sp: sp == "calib")
    E_test, y_test = _embed(lambda sp: sp.startswith("test"))

    # Per-source test embeddings, so we can show calibration transfers to
    # clean held-out instruments but degrades on Baskaran (the gate's job).
    test_splits = sorted({sp for sp in splits.values() if sp.startswith("test")})
    per_source = {sp.replace("test_", ""): _embed(lambda s, t=sp: s == t) for sp in test_splits}

    if len(E_calib) == 0:
        raise RuntimeError(f"split-mode {args.split_mode} has no calib split; "
                           "calibration needs held-out known data.")

    # --- fit prototypes (train) + temperature (calib) ---------------------
    protos = class_prototypes(E_train, y_train)
    d2_calib = squared_distances(E_calib, protos)
    T = fit_temperature(d2_calib, y_calib)

    # --- evaluate on the pooled held-out test -----------------------------
    d2_test = squared_distances(E_test, protos)
    probs_before = softmax_from_dist(d2_test, 1.0)
    probs_after = softmax_from_dist(d2_test, T)
    before = _metrics(probs_before, y_test, args.bins)
    after = _metrics(probs_after, y_test, args.bins)

    # Per-source calibration at T* (does the clean-calib temperature transfer?).
    per_source_metrics: dict[str, dict] = {}
    for name, (E_s, y_s) in per_source.items():
        m = _metrics(softmax_from_dist(squared_distances(E_s, protos), T), y_s, args.bins)
        per_source_metrics[name] = {"n": int(len(y_s)),
                                    **{k: m[k] for k in ("accuracy", "nll", "brier", "ece")}}

    # k-NN agreement: does the calibrated prototype argmax match the
    # established k-NN decision the headline accuracy is reported on?
    knn = KNeighborsClassifier(n_neighbors=args.k, metric="minkowski", p=2).fit(E_train, y_train)
    y_knn = knn.predict(E_test)
    proto_pred = probs_after.argmax(axis=1)
    agreement = float((proto_pred == y_knn).mean())
    knn_acc = float(accuracy_score(y_test, y_knn))

    # --- persist for reuse by the gate + fusion ---------------------------
    out_dir = Path(RUNS_DIR) / args.embedder_run / "closedset"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out_dir / "calibration.npz",
             prototypes=protos, temperature=np.array([T]),
             classes=np.array(POLYMER_CLASSES))
    summary = {
        "embedder_run": args.embedder_run,
        "split_mode":   args.split_mode,
        "checkpoint":   args.checkpoint,
        "temperature":  T,
        "n_calib":      int(len(y_calib)),
        "n_test":       int(len(y_test)),
        "knn_accuracy": knn_acc,
        "proto_knn_agreement": agreement,
        "before": {k: v for k, v in before.items() if k != "bins"},
        "after":  {k: v for k, v in after.items() if k != "bins"},
        "per_source_after": per_source_metrics,
    }
    write_json(out_dir / "summary.json", summary)
    save_reliability_diagram(before, after, T, out_dir / "reliability.png")

    # --- console report ---------------------------------------------------
    print(f"\nClosed-set calibration  (run={args.embedder_run}, split={args.split_mode})")
    print(f"  fitted temperature T*           {T:.3f}   (calib n={len(y_calib)})")
    print(f"  prototype vs k-NN agreement     {agreement:.1%}   (k-NN test acc {knn_acc:.1%})")
    print(f"\n  {'metric':<12} {'T=1':>10} {'T*':>10}")
    print("  " + "-" * 34)
    for key in ("accuracy", "nll", "brier", "ece"):
        fmt = ".1%" if key == "accuracy" else ".4f"
        print(f"  {key:<12} {format(before[key], fmt):>10} {format(after[key], fmt):>10}")

    print(f"\n  per-source calibration at T*  (does the clean-calib temperature transfer?)")
    print(f"  {'source':<14} {'n':>4} {'acc':>7} {'ece':>8} {'nll':>8} {'brier':>8}")
    print("  " + "-" * 52)
    for name, m in per_source_metrics.items():
        print(f"  {name:<14} {m['n']:>4} {m['accuracy']:>7.1%} {m['ece']:>8.4f} {m['nll']:>8.4f} {m['brier']:>8.4f}")

    print(f"\nReliability diagram + calibration.npz written to {out_dir}")


if __name__ == "__main__":
    main()
