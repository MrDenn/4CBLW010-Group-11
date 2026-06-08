"""Combined held-out evaluation across heterogeneous test sources.

Pools every honest test source the project has - the parquet test splits
(FLOPP-e, OpenSpecy) plus one or more folders of external .txt spectra
(our lab measurements) - into a single k-NN-on-embeddings evaluation, and
renders one combined confusion matrix + embedding map plus a per-source
grid so the sources can be compared side by side.

All spectra are classified by the same mechanism as predict.py /
evaluate.py: k-NN against the training-split embedding gallery.

Usage:
  python -m src.evaluate_combined --embedder-run _source_out_smoke \
      --split-mode source_out --samples "data/raw/Lab Combined"
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neighbors import KNeighborsClassifier

from src.config import CLASS_TO_IDX, IDX_TO_CLASS, RUNS_DIR
from src.data import SpectrumDataset, build_eval_loader, load_parquet, prepare_splits
from src.predict import (
    embed_external,
    label_from_filename,
    load_embedder,
    load_txt_spectrum,
    pct_T_to_absorbance,
    resample_to_canonical,
)
from src.train import embed_all
from src.utils import get_device, write_json
from src.viz import compute_projection, save_confusion_grid, save_confusion_matrix, save_embedding_map


def bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_boot: int,
    ci: float,
    seed: int,
) -> dict[str, dict[str, float]]:
    """Percentile bootstrap CIs for accuracy and macro-F1.

    Resamples the (true, pred) pairs with replacement n_boot times and
    recomputes each metric, then takes the central `ci` percentile band.
    This quantifies how much a metric could move just from the luck of
    which spectra landed in this (often small) test source - the honest
    error bar to report alongside the point estimate.

    macro-F1 is scored over a fixed label set (the classes present in the
    full sample) with zero_division=0, so a class dropping out of a
    resample lowers the score rather than silently changing the average's
    denominator.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = len(y_true)
    labels = sorted(set(y_true.tolist()) | set(y_pred.tolist()))
    rng = np.random.default_rng(seed)

    accs = np.empty(n_boot, dtype=np.float64)
    f1s = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yt, yp = y_true[idx], y_pred[idx]
        accs[b] = accuracy_score(yt, yp)
        f1s[b] = f1_score(yt, yp, labels=labels, average="macro", zero_division=0)

    lo_q = 100.0 * (1.0 - ci) / 2.0
    hi_q = 100.0 * (1.0 + ci) / 2.0
    return {
        "accuracy": {
            "lo": float(np.percentile(accs, lo_q)),
            "hi": float(np.percentile(accs, hi_q)),
        },
        "macro_f1": {
            "lo": float(np.percentile(f1s, lo_q)),
            "hi": float(np.percentile(f1s, hi_q)),
        },
    }


def _embed_txt_folder(model, folder: Path, device) -> tuple[np.ndarray, np.ndarray]:
    """Parse + preprocess + embed a folder of labelled .txt spectra.

    Only files whose name starts with a known polymer class are kept (we
    need ground-truth labels to score). Returns (embeddings, labels).
    """
    intensities, labels = [], []
    for f in sorted(folder.glob("*.txt")):
        true = label_from_filename(f.stem)
        if true is None:
            continue
        wn, y, y_units = load_txt_spectrum(f)
        if wn.size < 50:
            continue
        absb = pct_T_to_absorbance(y) if "T" in y_units else y.astype(np.float32)
        grid = resample_to_canonical(wn, absb)
        if grid is None:
            continue
        intensities.append(grid)
        labels.append(CLASS_TO_IDX[true])
    if not intensities:
        raise RuntimeError(f"No usable labelled .txt spectra in {folder}")
    E = embed_external(model, np.stack(intensities), device)
    return E, np.asarray(labels)


def main() -> None:
    p = argparse.ArgumentParser(description="Combined evaluation across all test sources.")
    p.add_argument("--embedder-run", required=True)
    p.add_argument("--split-mode", choices=("random", "source_out", "source_cv", "source_cv2"), default="source_out")
    p.add_argument("--checkpoint", default="best.pt")
    p.add_argument("--samples", action="append", default=[],
                   help="Folder of labelled .txt spectra (repeatable).")
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-boot", type=int, default=1000,
                   help="Bootstrap resamples for confidence intervals (0 disables).")
    p.add_argument("--ci", type=float, default=0.95,
                   help="Confidence level for the bootstrap intervals.")
    args = p.parse_args()

    device = get_device()
    model = load_embedder(args.embedder_run, args.checkpoint, device)

    # Gallery = training split embeddings.
    splits = prepare_splits(seed=args.seed, mode=args.split_mode)
    df = load_parquet()
    train_ids = [s for s, sp in splits.items() if sp == "train"]
    E_train, y_train = embed_all(model, build_eval_loader(SpectrumDataset(df, train_ids), 128), device)

    knn = KNeighborsClassifier(n_neighbors=args.k, metric="minkowski", p=2)
    knn.fit(E_train, y_train)

    # Collect every test source: parquet test splits + external folders.
    sources: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for sp in sorted({v for v in splits.values() if v.startswith("test")}):
        ids = [s for s, s_sp in splits.items() if s_sp == sp]
        E, y = embed_all(model, build_eval_loader(SpectrumDataset(df, ids), 128), device)
        sources[sp.replace("test_", "")] = (E, y)
    for folder in args.samples:
        fp = Path(folder)
        sources[fp.name] = _embed_txt_folder(model, fp, device)

    # Per-source + combined scoring.
    out_dir = Path(RUNS_DIR) / args.embedder_run / "figures_combined_all"
    out_dir.mkdir(parents=True, exist_ok=True)

    panels: list[tuple[str, list[int], list[int]]] = []
    summary: dict[str, dict] = {}
    E_all, y_all = [], []
    for name, (E, y) in sources.items():
        y_pred = knn.predict(E)
        panels.append((name, list(y), list(y_pred)))
        summary[name] = {
            "n": int(len(y)),
            "accuracy": float(accuracy_score(y, y_pred)),
            "macro_f1": float(f1_score(y, y_pred, average="macro")),
        }
        if args.n_boot > 0:
            summary[name]["ci"] = bootstrap_ci(y, y_pred, args.n_boot, args.ci, args.seed)
        E_all.append(E); y_all.append(y)

    E_all = np.concatenate(E_all); y_all = np.concatenate(y_all)
    y_pred_all = knn.predict(E_all)
    summary["COMBINED"] = {
        "n": int(len(y_all)),
        "accuracy": float(accuracy_score(y_all, y_pred_all)),
        "macro_f1": float(f1_score(y_all, y_pred_all, average="macro")),
    }
    if args.n_boot > 0:
        summary["COMBINED"]["ci"] = bootstrap_ci(y_all, y_pred_all, args.n_boot, args.ci, args.seed)
    panels.append(("all sources combined", list(y_all), list(y_pred_all)))

    # Figures.
    save_confusion_matrix(
        list(y_all), list(y_pred_all), out_dir / "confusion_combined.png",
        title=f"All test sources combined\n"
              f"accuracy = {summary['COMBINED']['accuracy']:.1%}  (n = {len(y_all)})",
    )
    G2, X2 = compute_projection(E_train, E_all, k=args.k)
    save_embedding_map(
        G2, X2, y_train, list(y_all), list(y_pred_all),
        out_dir / "embedding_map_combined.png",
        title="All test sources combined in the model's embedding space",
        sample_label="test spectrum",
    )
    save_confusion_grid(panels, out_dir / "confusion_grid.png",
                        suptitle="Per-source and combined - k-NN on embeddings")

    write_json(out_dir / "summary.json", summary)

    # Console report.
    if args.n_boot > 0:
        pct = int(round(args.ci * 100))
        print(f"\n{'source':<22} {'n':>5} {'acc':>7}  {f'{pct}% CI':>15} "
              f"{'macroF1':>8}  {f'{pct}% CI':>15}")
        print("-" * 78)
        for name, m in summary.items():
            ci = m["ci"]
            acc_ci = f"[{ci['accuracy']['lo']:.1%}, {ci['accuracy']['hi']:.1%}]"
            f1_ci = f"[{ci['macro_f1']['lo']:.3f}, {ci['macro_f1']['hi']:.3f}]"
            print(f"{name:<22} {m['n']:>5} {m['accuracy']:>7.1%}  {acc_ci:>15} "
                  f"{m['macro_f1']:>8.3f}  {f1_ci:>15}")
    else:
        print(f"\n{'source':<24} {'n':>5} {'acc':>8} {'macroF1':>9}")
        print("-" * 48)
        for name, m in summary.items():
            print(f"{name:<24} {m['n']:>5} {m['accuracy']:>8.1%} {m['macro_f1']:>9.3f}")
    print(f"\nFigures + summary written to {out_dir}")


if __name__ == "__main__":
    main()
