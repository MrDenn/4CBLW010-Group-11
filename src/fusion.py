"""Stage 3 — fusion: the unified 7-way confidence output.

Stage 1 (`src/closedset.py`) gives P(class | known) — which of the six,
how sure, calibrated. Stage 2 (`src/gate.py`) gives a novelty score —
how far from everything the model knows. This module joins them into a
single distribution over **7 outcomes** (the six plastics + "unknown"):

    p_known     = P(this is one of the six at all)        (from the gate)
    P(class_i)  = p_known * P(class_i | known)             (six entries)
    P(unknown)  = 1 - p_known

By the law of total probability this sums to 1, and the unknown mass
*discounts* the six class confidences — a confident closed-set call on a
likely-OOD spectrum is correctly pulled down. The system abstains when
P(unknown) exceeds the operating threshold.

p_known is obtained by calibrating the novelty score on a **cross-
instrument known reference** (per gate.py Finding 2, held-in calib does
not transfer). We calibrate on the CLEAN held-out instruments
(OpenSpecy, FLOPP-e, In-house) and then *deploy* on (a) those instruments
via leave-one-out, (b) Baskaran — a never-seen degraded-multilayer stream,
and (c) the 823 open OOD spectra. That is the real deployment question:
calibrate on normal big-six instruments, behave correctly on a new
degraded stream and on non-plastics.

Usage:
  python -m src.fusion --embedder-run cv2_aug \
      --split-mode source_cv2 --checkpoint latest.pt --accept 0.90
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless

import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import NearestNeighbors

from src.closedset import class_prototypes, fit_temperature, softmax_from_dist, squared_distances
from src.config import IDX_TO_CLASS, NUM_CLASSES, POLYMER_CLASSES, RUNS_DIR
from src.data import SpectrumDataset, build_eval_loader, load_parquet, prepare_splits
from src.gate import knn_distance, load_open
from src.predict import embed_external, load_embedder
from src.train import embed_all
from src.utils import get_device, write_json

CLEAN_INSTRUMENTS = ("OpenSpecy", "FLOPP-e", "In-house")  # calibration reference


# ---------------------------------------------------------------------------
# p_known and fusion
# ---------------------------------------------------------------------------


def fit_pknown(s_ref: np.ndarray):
    """Knowns-only map novelty score -> p_known (monotone decreasing).

    Conformal-style survival function: p_known(s) = fraction of reference
    knowns whose novelty is >= s (with +1 smoothing). A spectrum more
    novel than every known -> p_known ~ 0; one typical of knowns -> ~1.
    Uses NO unknowns, so the gate cannot overfit to the OOD set.
    """
    s_sorted = np.sort(s_ref)
    n = len(s_sorted)

    def pknown(s: np.ndarray) -> np.ndarray:
        s = np.asarray(s, dtype=np.float64)
        below = np.searchsorted(s_sorted, s, side="left")  # #ref strictly < s
        count_ge = n - below
        return (count_ge + 1.0) / (n + 1.0)

    return pknown


def fuse(closed_probs: np.ndarray, p_known: np.ndarray) -> np.ndarray:
    """Unified 7-way distribution; last column is 'unknown'. Rows sum to 1."""
    out = np.empty((len(p_known), NUM_CLASSES + 1), dtype=np.float64)
    out[:, :NUM_CLASSES] = p_known[:, None] * closed_probs
    out[:, NUM_CLASSES] = 1.0 - p_known
    return out


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------


def save_outcome_bars(rows: dict[str, dict], out: Path) -> None:
    """Per-source stacked bars: accept-correct / accept-wrong / abstain."""
    names = list(rows)
    correct = [rows[n]["accept_correct_frac"] for n in names]
    wrong = [rows[n]["accept_wrong_frac"] for n in names]
    abstain = [rows[n]["abstain_frac"] for n in names]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(names, correct, color="#2e7d32", label="accepted & correct")
    ax.bar(names, wrong, bottom=correct, color="#c62828", label="accepted & wrong")
    ax.bar(names, abstain, bottom=np.add(correct, wrong), color="#9e9e9e",
           label="abstained (P(unknown) high)")
    ax.set_ylabel("fraction of spectra")
    ax.set_ylim(0, 1)
    ax.set_title("Fused 7-way head at the operating point\n"
                 "knowns: accepted & mostly correct · Baskaran/open: abstained")
    ax.legend(loc="lower right")
    plt.xticks(rotation=20, ha="right")
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def _fmt_vec(v: np.ndarray) -> str:
    labels = list(POLYMER_CLASSES) + ["UNKNOWN"]
    top = np.argsort(v)[::-1][:3]
    return "  ".join(f"{labels[i]}={v[i]:.2f}" for i in top)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage-3 fusion: unified 7-way confidence output.")
    p.add_argument("--embedder-run", required=True)
    p.add_argument("--split-mode", choices=("random", "source_out", "source_cv", "source_cv2"),
                   default="source_cv2")
    p.add_argument("--checkpoint", default="latest.pt")
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--accept", type=float, default=0.90,
                   help="Target acceptance rate on the clean known reference (sets the threshold).")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    return args


def main() -> None:
    args = parse_args()
    device = get_device()
    model = load_embedder(args.embedder_run, args.checkpoint, device)
    splits = prepare_splits(seed=args.seed, mode=args.split_mode)
    df = load_parquet()

    def _embed(pred) -> tuple[np.ndarray, np.ndarray]:
        ids = [s for s, sp in splits.items() if pred(sp)]
        return embed_all(model, build_eval_loader(SpectrumDataset(df, ids), 128), device)

    E_train, y_train = _embed(lambda sp: sp == "train")
    E_calib, y_calib = _embed(lambda sp: sp == "calib")
    test_splits = sorted({sp for sp in splits.values() if sp.startswith("test")})
    test_by_src = {sp.replace("test_", ""): _embed(lambda s, t=sp: s == t) for sp in test_splits}

    open_df = load_open()
    E_open = embed_external(model, np.stack(open_df["intensity"].to_numpy()), device)
    open_cats = open_df["category"].to_numpy()

    # Stage 1 pieces.
    protos = class_prototypes(E_train, y_train)
    T = fit_temperature(squared_distances(E_calib, protos), y_calib)
    # Stage 2 scorer.
    nn = NearestNeighbors(n_neighbors=args.k).fit(E_train)

    def closed(E):
        return softmax_from_dist(squared_distances(E, protos), T)

    def novelty(E):
        return knn_distance(nn, E)

    # Pre-compute per-source novelty + closed probs + the clean-reference scores.
    src_novelty = {n: novelty(E) for n, (E, _) in test_by_src.items()}
    clean_present = [c for c in CLEAN_INSTRUMENTS if c in test_by_src]
    s_clean_all = np.concatenate([src_novelty[c] for c in clean_present])

    # --- per-source evaluation at the operating point ---------------------
    # Knowns: calibrate on the OTHER clean instruments (leave-one-out among
    # clean). Baskaran / non-clean: calibrate on ALL clean instruments
    # (never-seen deployment). Open: same all-clean reference.
    rows: dict[str, dict] = {}
    fused_store: dict[str, np.ndarray] = {}
    for name, (E, y) in test_by_src.items():
        if name in clean_present:
            ref = np.concatenate([src_novelty[c] for c in clean_present if c != name])
        else:
            ref = s_clean_all  # e.g. Baskaran: full clean reference, never-seen
        tau = float(np.quantile(ref, args.accept))
        pk = fit_pknown(ref)
        s = src_novelty[name]
        fused = fuse(closed(E), pk(s))
        fused_store[name] = fused
        accept = s <= tau
        pred = closed(E).argmax(axis=1)
        correct = (pred == y)
        n = len(y)
        acc_correct = int((accept & correct).sum())
        acc_wrong = int((accept & ~correct).sum())
        rows[name] = {
            "n": n,
            "tau": tau,
            "coverage": float(accept.mean()),
            "abstain_frac": float((~accept).mean()),
            "accept_correct_frac": acc_correct / n,
            "accept_wrong_frac": acc_wrong / n,
            "accuracy_on_accepted": float(correct[accept].mean()) if accept.any() else float("nan"),
            "full_accuracy": float(correct.mean()),
        }

    # Open (OOD): all-clean reference; correct behaviour = abstain.
    tau_open = float(np.quantile(s_clean_all, args.accept))
    pk_open = fit_pknown(s_clean_all)
    s_open = novelty(E_open)
    fused_open = fuse(closed(E_open), pk_open(s_open))
    open_accept = s_open <= tau_open
    rows["open (OOD)"] = {
        "n": int(len(s_open)),
        "tau": tau_open,
        "coverage": float(open_accept.mean()),          # = wrongly accepted
        "abstain_frac": float((~open_accept).mean()),   # = correctly rejected
        "accept_correct_frac": 0.0,                     # no correct accept possible
        "accept_wrong_frac": float(open_accept.mean()),
        "accuracy_on_accepted": float("nan"),
        "full_accuracy": float("nan"),
    }
    per_cat_reject = {c: float((s_open[open_cats == c] > tau_open).mean())
                      for c in sorted(set(open_cats))}

    # --- persist + figure -------------------------------------------------
    out_dir = Path(RUNS_DIR) / args.embedder_run / "fusion"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "embedder_run": args.embedder_run, "split_mode": args.split_mode,
        "accept_target": args.accept, "temperature": T,
        "calibration_reference": clean_present,
        "per_source": rows, "per_category_open_reject": per_cat_reject,
    }
    write_json(out_dir / "summary.json", summary)
    save_outcome_bars(rows, out_dir / "outcome_bars.png")

    # --- console report ---------------------------------------------------
    print(f"\nStage-3 fusion — unified 7-way output  (run={args.embedder_run})")
    print(f"  P(class)=p_known*P(class|known), P(unknown)=1-p_known")
    print(f"  calibrated on clean instruments {clean_present}; accept target {args.accept:.0%}\n")
    print(f"  {'source':<14} {'n':>4} {'coverage':>9} {'acc|accept':>11} {'abstain':>8}")
    print("  " + "-" * 50)
    for name, m in rows.items():
        acc = "  n/a  " if m["accuracy_on_accepted"] != m["accuracy_on_accepted"] else f"{m['accuracy_on_accepted']:.1%}"
        print(f"  {name:<14} {m['n']:>4} {m['coverage']:>9.1%} {acc:>11} {m['abstain_frac']:>8.1%}")

    print(f"\n  Baskaran (degraded multilayer, never in calibration):")
    b = rows.get("Baskaran")
    if b:
        print(f"    abstains on {b['abstain_frac']:.1%}; of the {b['coverage']:.1%} it accepts, "
              f"{b['accuracy_on_accepted']:.1%} are correct (full-set accuracy was {b['full_accuracy']:.1%})")

    print(f"\n  open (OOD) correctly rejected per category:")
    for c, r in sorted(per_cat_reject.items(), key=lambda kv: -kv[1]):
        print(f"    {c:<20} {r:.1%}")

    # Example unified outputs (the deliverable, made concrete).
    print(f"\n  Example fused outputs (top-3 of the 7-way distribution):")
    examples = []
    if clean_present:
        c0 = clean_present[0]
        fc = fused_store[c0]
        examples.append((f"{c0} most confident", fc[fc[:, :NUM_CLASSES].max(1).argmax()]))
    if "Baskaran" in fused_store:
        fb = fused_store["Baskaran"]
        examples.append(("Baskaran most-novel", fb[fb[:, NUM_CLASSES].argmax()]))
    examples.append(("open most-novel", fused_open[fused_open[:, NUM_CLASSES].argmax()]))
    for label, vec in examples:
        print(f"    {label:<22} -> {_fmt_vec(vec)}")

    print(f"\nSummary + figure written to {out_dir}")


if __name__ == "__main__":
    main()
