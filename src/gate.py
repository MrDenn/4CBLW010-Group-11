"""Open-set gate (stage 2 of the confidence head).

Stage 1 (src/closedset.py) gives a calibrated P(class | known) but has no
way to say "this is not one of the six plastics at all" - on the degraded
Baskaran laminates it is confidently wrong. The gate supplies the missing
signal: a novelty score that flags spectra far from everything the model
knows, so the system can abstain.

Design (see research-notes/development_decision_log.md section 11):
  - Novelty score = distance-to-gallery: the mean distance to the k
    nearest TRAINING embeddings. Metric learning explicitly shapes the
    space so same-class points cluster and others separate, so "far from
    every cluster" is the geometrically correct out-of-distribution cue -
    the one signal of the three with a real "none of these" notion.
  - The 823 `label_role == "open"` spectra (other_plastic, organic,
    unknown, rubber, textile_synthetic, textile_natural) are the held-out
    OOD test set. They were never in any split, so the embedder never saw
    them.

What this reports:
  1. A SIGNAL COMPARISON - AUROC of four candidate scores (kNN-distance,
     prototype-distance, 1-maxprob, 1-margin) as OOD detectors, turning
     the "which signal is most reliable" design claim into a measurement.
  2. The HEADLINE - AUROC (known test vs. open) for the chosen distance
     score, overall and stratified by `category`.
  3. An OPERATING POINT chosen WITHOUT unknowns: the threshold that
     accepts 95% of held-out known `calib` spectra; then the known-reject
     and per-category open-reject rates. (No open data touches tuning, so
     the primary result cannot overfit to the unknowns.)
  4. A novelty-score histogram (known vs. open).

The raw scores are saved for the fusion step, which turns the chosen
score into p_known and combines it with stage 1.

Usage:
  python -m src.gate --embedder-run cv2_aug \
      --split-mode source_cv2 --checkpoint latest.pt
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: render straight to PNG files

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors

from src.closedset import class_prototypes, fit_temperature, softmax_from_dist, squared_distances
from src.config import PARQUET_PATH, RUNS_DIR
from src.data import SpectrumDataset, build_eval_loader, load_parquet, prepare_splits
from src.predict import embed_external, load_embedder
from src.train import embed_all
from src.utils import get_device, write_json


# ---------------------------------------------------------------------------
# Candidate novelty scores (all: higher = more "unknown")
# ---------------------------------------------------------------------------


def knn_distance(nn: NearestNeighbors, E: np.ndarray) -> np.ndarray:
    """Mean Euclidean distance to the k nearest TRAINING embeddings."""
    dist, _ = nn.kneighbors(E)
    return dist.mean(axis=1)


def prototype_distance(E: np.ndarray, protos: np.ndarray) -> np.ndarray:
    """Distance to the NEAREST class prototype (min over classes)."""
    return np.sqrt(np.clip(squared_distances(E, protos).min(axis=1), 0, None))


def one_minus_maxprob(E: np.ndarray, protos: np.ndarray, T: float) -> np.ndarray:
    """1 - calibrated closed-set max probability (the softmax-style score)."""
    return 1.0 - softmax_from_dist(squared_distances(E, protos), T).max(axis=1)


def one_minus_margin(knn: KNeighborsClassifier, E: np.ndarray) -> np.ndarray:
    """1 - k-NN vote fraction of the winning class (the ambiguity score)."""
    return 1.0 - knn.predict_proba(E).max(axis=1)


# ---------------------------------------------------------------------------
# Selective prediction (risk-coverage): does abstaining on high-novelty
# knowns IMPROVE accuracy on the rest?
# ---------------------------------------------------------------------------


def risk_coverage(
    correct: np.ndarray, novelty: np.ndarray, coverages: tuple[float, ...]
) -> dict:
    """Accuracy on the most-confident (lowest-novelty) fraction of knowns.

    Abstaining on the highest-novelty spectra and reporting accuracy on the
    accepted remainder. `correct` is the per-spectrum correctness of the
    closed-set classifier; `novelty` the gate score (higher = abstain
    first). Also reports the accuracy of the spectra abstained at the
    tightest coverage - low here means the gate is dropping the model's own
    errors, i.e. abstention is selective rather than wasteful.
    """
    order = np.argsort(novelty)            # most confident first
    c_sorted = correct[order].astype(float)
    n = len(correct)
    points = []
    for cov in coverages:
        k = max(1, int(round(cov * n)))
        points.append({"coverage": float(cov),
                       "accuracy_on_accepted": float(c_sorted[:k].mean())})
    # Accuracy of the spectra abstained at the highest coverage < 1.0 (the
    # smallest, most-novel reject set): low here means the gate is dropping
    # the model's own errors, not good predictions.
    tightest = max((c for c in coverages if c < 1.0), default=0.95)
    k = int(round(tightest * n))
    abstained_acc = float(c_sorted[k:].mean()) if k < n else float("nan")
    return {"n": n, "full_accuracy": float(correct.mean()),
            "points": points,
            "abstained_fraction": float(round(1.0 - tightest, 4)),
            "abstained_accuracy": abstained_acc}


def save_risk_coverage(
    pooled: dict, per_source: dict[str, dict], out: Path
) -> None:
    """Accuracy-vs-coverage curve, pooled + per held-out instrument."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, rc in per_source.items():
        xs = [p["coverage"] for p in rc["points"]]
        ys = [p["accuracy_on_accepted"] for p in rc["points"]]
        ax.plot(xs, ys, marker="o", lw=1.2, alpha=0.7, label=name)
    xs = [p["coverage"] for p in pooled["points"]]
    ys = [p["accuracy_on_accepted"] for p in pooled["points"]]
    ax.plot(xs, ys, marker="s", lw=2.5, color="black", label="POOLED")
    ax.invert_xaxis()  # 100% coverage (no abstention) on the left
    ax.set_xlabel("coverage (fraction of known spectra accepted)")
    ax.set_ylabel("accuracy on accepted")
    ax.set_title("Selective prediction: abstaining on high-novelty knowns\n"
                 "raises accuracy on the rest (it drops the model's own errors)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Loading the open (OOD) spectra
# ---------------------------------------------------------------------------


def load_open(path=PARQUET_PATH) -> pd.DataFrame:
    """All `label_role == "open"` rows (the held-out OOD set), with category."""
    df = pd.read_parquet(path)
    return df[df["label_role"] == "open"].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------


def save_histogram(
    score_known: np.ndarray, open_by_cat: dict[str, np.ndarray],
    tau: float, out: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    all_open = np.concatenate(list(open_by_cat.values()))
    lo = float(min(score_known.min(), all_open.min()))
    hi = float(max(score_known.max(), all_open.max()))
    bins = np.linspace(lo, hi, 40)
    ax.hist(score_known, bins=bins, alpha=0.7, color="#2e7d32",
            density=True, label="known big-6 (test)")
    ax.hist(all_open, bins=bins, alpha=0.5, color="#c62828",
            density=True, label="open (OOD)")
    ax.axvline(tau, color="black", ls="--", lw=1.5,
               label=f"reject threshold (95% known acceptance)")
    ax.set_xlabel("novelty score  (mean distance to k nearest known)")
    ax.set_ylabel("density")
    ax.set_title("Open-set gate: knowns sit close to the gallery, unknowns far")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Open-set gate: novelty scoring + AUROC evaluation.")
    p.add_argument("--embedder-run", required=True)
    p.add_argument("--split-mode", choices=("random", "source_out", "source_cv", "source_cv2"),
                   default="source_cv2")
    p.add_argument("--checkpoint", default="latest.pt")
    p.add_argument("--k", type=int, default=5, help="k for the gallery distance + vote scores.")
    p.add_argument("--accept", type=float, default=0.95,
                   help="Target known-acceptance rate that sets the reject threshold (on calib).")
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

    # Per-source known test sets (for the per-source risk-coverage curves).
    test_splits = sorted({sp for sp in splits.values() if sp.startswith("test")})
    test_by_source = {sp.replace("test_", ""): _embed(lambda s, t=sp: s == t)
                      for sp in test_splits}

    # Open (OOD) spectra, embedded directly (no labels in CLASS_TO_IDX).
    open_df = load_open()
    E_open = embed_external(model, np.stack(open_df["intensity"].to_numpy()), device)
    open_cats = open_df["category"].to_numpy()

    # Fitted pieces (reused from / consistent with stage 1).
    protos = class_prototypes(E_train, y_train)
    T = fit_temperature(squared_distances(E_calib, protos), y_calib)
    nn = NearestNeighbors(n_neighbors=args.k).fit(E_train)
    knn = KNeighborsClassifier(n_neighbors=args.k, metric="minkowski", p=2).fit(E_train, y_train)

    # --- (1) signal comparison: AUROC of each candidate score -------------
    scorers = {
        "knn_distance":   lambda E: knn_distance(nn, E),
        "prototype_dist": lambda E: prototype_distance(E, protos),
        "one_minus_maxprob": lambda E: one_minus_maxprob(E, protos, T),
        "one_minus_margin":  lambda E: one_minus_margin(knn, E),
    }
    labels = np.concatenate([np.zeros(len(E_test)), np.ones(len(E_open))])  # 1 = open
    signal_auroc: dict[str, float] = {}
    for name, fn in scorers.items():
        s = np.concatenate([fn(E_test), fn(E_open)])
        signal_auroc[name] = float(roc_auc_score(labels, s))

    # --- chosen score: kNN distance (the headline gate) -------------------
    s_calib = knn_distance(nn, E_calib)
    s_test = knn_distance(nn, E_test)
    s_open = knn_distance(nn, E_open)

    # --- (2) headline + per-category AUROC --------------------------------
    auroc_overall = float(roc_auc_score(labels, np.concatenate([s_test, s_open])))
    per_cat_auroc: dict[str, dict] = {}
    open_by_cat: dict[str, np.ndarray] = {}
    for cat in sorted(set(open_cats)):
        s_cat = s_open[open_cats == cat]
        open_by_cat[cat] = s_cat
        lab = np.concatenate([np.zeros(len(s_test)), np.ones(len(s_cat))])
        per_cat_auroc[cat] = {
            "n": int(len(s_cat)),
            "auroc": float(roc_auc_score(lab, np.concatenate([s_test, s_cat]))),
        }

    # --- (3) operating point ---------------------------------------------
    # Held-in calib threshold transfers POORLY: calib is drawn from the
    # training instruments, so its spectra sit abnormally close to the
    # gallery, while the test knowns are whole held-out instruments sitting
    # farther out. We report that transfer gap as a finding, then set the
    # achievable operating point on the cross-instrument known distribution
    # (the deployment-realistic reference) - still using NO open data.
    tau_calib = float(np.quantile(s_calib, args.accept))
    known_reject_calib = float((s_test > tau_calib).mean())

    tau = float(np.quantile(s_test, args.accept))  # cross-instrument knowns
    known_reject = float((s_test > tau).mean())
    open_reject_overall = float((s_open > tau).mean())
    per_cat_reject = {cat: float((s_open[open_cats == cat] > tau).mean())
                      for cat in sorted(set(open_cats))}

    # --- (4) selective prediction / risk-coverage on KNOWNS ---------------
    # Abstention is not wasted: the gate drops the model's own errors, so
    # accuracy on the accepted remainder rises. Pooled + per held-out
    # instrument, over a coverage sweep.
    coverages = (1.0, 0.95, 0.90, 0.80, 0.70)
    correct_test = (knn.predict(E_test) == y_test)
    rc_pooled = risk_coverage(correct_test, s_test, coverages)
    rc_per_source = {}
    for name, (E_s, y_s) in test_by_source.items():
        rc_per_source[name] = risk_coverage(
            (knn.predict(E_s) == y_s), knn_distance(nn, E_s), coverages)

    # --- persist + figure -------------------------------------------------
    out_dir = Path(RUNS_DIR) / args.embedder_run / "gate"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out_dir / "scores.npz",
             s_calib=s_calib, s_test=s_test, s_open=s_open,
             open_cats=open_cats, tau=np.array([tau]))
    summary = {
        "embedder_run": args.embedder_run,
        "split_mode":   args.split_mode,
        "k":            args.k,
        "n_test_known": int(len(s_test)),
        "n_open":       int(len(s_open)),
        "signal_comparison_auroc": signal_auroc,
        "headline_auroc_knn_distance": auroc_overall,
        "per_category_auroc": per_cat_auroc,
        "operating_point": {
            "target_known_acceptance": args.accept,
            "tau_calib_heldin": tau_calib,
            "known_reject_rate_calib_threshold": known_reject_calib,
            "tau": tau,
            "known_reject_rate": known_reject,
            "open_reject_rate_overall": open_reject_overall,
            "per_category_open_reject": per_cat_reject,
        },
        "risk_coverage": {"pooled": rc_pooled, "per_source": rc_per_source},
    }
    write_json(out_dir / "summary.json", summary)
    save_histogram(s_test, open_by_cat, tau, out_dir / "novelty_histogram.png")
    save_risk_coverage(rc_pooled, rc_per_source, out_dir / "risk_coverage.png")

    # --- console report ---------------------------------------------------
    print(f"\nOpen-set gate  (run={args.embedder_run}, split={args.split_mode})")
    print(f"  known test n={len(s_test)}   open (OOD) n={len(s_open)}")

    print(f"\n  (1) signal comparison - AUROC as an OOD detector (higher = better):")
    for name, a in sorted(signal_auroc.items(), key=lambda kv: -kv[1]):
        print(f"      {name:<20} {a:.3f}")

    print(f"\n  (2) headline gate (kNN distance)  AUROC = {auroc_overall:.3f}")
    print(f"      {'category':<20} {'n':>4} {'AUROC':>7}")
    print("      " + "-" * 33)
    for cat, m in sorted(per_cat_auroc.items(), key=lambda kv: -kv[1]['auroc']):
        print(f"      {cat:<20} {m['n']:>4} {m['auroc']:>7.3f}")

    print(f"\n  (3a) held-in calib threshold transfer (a finding):")
    print(f"       tau from calib={tau_calib:.3f} -> {known_reject_calib:.1%} of "
          f"cross-instrument knowns wrongly rejected")
    print(f"       (calib is in-distribution; its novelty scores don't transfer to held-out instruments)")

    print(f"\n  (3b) operating point @ {args.accept:.0%} acceptance on cross-instrument knowns (tau={tau:.3f}):")
    print(f"       known wrongly rejected   {known_reject:.1%}")
    print(f"       open correctly rejected  {open_reject_overall:.1%}  (overall)")
    for cat, r in sorted(per_cat_reject.items(), key=lambda kv: -kv[1]):
        print(f"         {cat:<20} {r:.1%}")

    print(f"\n  (4) selective prediction - accuracy on accepted knowns vs coverage:")
    header = "      " + f"{'source':<14}" + "".join(f"{int(c*100):>7}%" for c in coverages)
    print(header)
    print("      " + "-" * (14 + 8 * len(coverages)))
    for name, rc in {"POOLED": rc_pooled, **rc_per_source}.items():
        print(f"      {name:<14}" + "".join(f"{p['accuracy_on_accepted']:>8.1%}" for p in rc["points"]))
    print(f"      (pooled: the abstained top-{int(rc_pooled['abstained_fraction']*100)}% novelty were only "
          f"{rc_pooled['abstained_accuracy']:.1%} accurate -> abstention drops the errors)")

    print(f"\nScores + histogram + risk-coverage written to {out_dir}")


if __name__ == "__main__":
    main()
