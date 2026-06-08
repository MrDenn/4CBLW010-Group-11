"""Classify external FTIR .txt spectra with a trained embedder.

Point this at a folder of two-column (wavenumber, intensity) text files -
e.g. JCAMP-style exports straight off an FTIR instrument - and it will:

  1. parse each .txt (skipping the ## header), detecting %T vs absorbance
  2. convert %T -> absorbance and resample onto the canonical 882-grid,
     i.e. the exact transform compile_data.py applies to training data
  3. embed every spectrum with a trained SmolenCNN checkpoint
  4. classify each by k-NN against the training-set embedding gallery
  5. if filenames start with a known polymer (e.g. "HDPE 3.txt"), report
     accuracy / macro-F1 / a confusion matrix
  6. save slide-ready figures: confusion matrix, per-sample results,
     and a UMAP map of where the new samples land among the training data

Usage:
  python -m src.predict --samples "data/raw/Denis Fenne 21.05" \
      --embedder-run baseline_multisim --split-mode source_out
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: render straight to PNG files

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.interpolate import interp1d
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neighbors import KNeighborsClassifier

from src.config import (
    CANONICAL_HI,
    CANONICAL_LO,
    CLASS_TO_IDX,
    IDX_TO_CLASS,
    INPUT_LEN_RAW,
    RUNS_DIR,
)
from src.data import (
    SpectrumDataset,
    build_eval_loader,
    load_parquet,
    prepare_splits,
    preprocess_spectra,
)
from src.viz import (
    compute_projection,
    draw_confusion,
    draw_embedding_map,
    save_confusion_matrix,
    save_embedding_map,
)
from src.model import SmolenCNN
from src.train import embed_all
from src.utils import get_device, load_checkpoint, write_json

# Canonical wavenumber grid - must match compile_data.py exactly.
CANONICAL_WN = np.linspace(CANONICAL_LO, CANONICAL_HI, INPUT_LEN_RAW).astype(np.float32)


# ---------------------------------------------------------------------------
# Parsing external .txt spectra
# ---------------------------------------------------------------------------


def load_txt_spectrum(path: Path) -> tuple[np.ndarray, np.ndarray, str]:
    """Parse a JCAMP-style two-column FTIR text file.

    Returns (wavenumbers, intensities, y_units). Header lines start with
    `##`; `##YUNITS=` tells us whether the y column is %T or absorbance.
    """
    y_units = "UNKNOWN"
    wn: list[float] = []
    y: list[float] = []
    for line in path.read_text(errors="ignore").splitlines():
        s = line.strip()
        if not s:
            continue
        if s.startswith("##"):
            if s.upper().startswith("##YUNITS"):
                y_units = s.split("=", 1)[-1].strip().upper()
            continue
        parts = s.replace(",", " ").split()
        if len(parts) >= 2:
            try:
                wn.append(float(parts[0]))
                y.append(float(parts[1]))
            except ValueError:
                continue
    return np.asarray(wn, dtype=np.float32), np.asarray(y, dtype=np.float32), y_units


def pct_T_to_absorbance(y_pct: np.ndarray) -> np.ndarray:
    """%-transmittance -> absorbance. Mirrors compile_data.py."""
    T = np.clip(y_pct.astype(np.float32) / 100.0, 1e-4, None)
    return (-np.log10(T)).astype(np.float32)


def resample_to_canonical(wn: np.ndarray, y: np.ndarray) -> np.ndarray | None:
    """Linearly resample (wn, y) onto the canonical 882-grid.

    Returns None if the spectrum does not cover enough of the grid range
    (same coverage rule as compile_data.py).
    """
    order = np.argsort(wn)
    wn, y = wn[order], y[order]
    wn, idx = np.unique(wn, return_index=True)
    y = y[idx]
    if wn.min() > CANONICAL_LO + 100 or wn.max() < CANONICAL_HI - 100:
        return None
    f = interp1d(
        wn, y, kind="linear", bounds_error=False,
        fill_value=(float(np.median(y[:5])), float(np.median(y[-5:]))),
    )
    return f(CANONICAL_WN).astype(np.float32)


def label_from_filename(stem: str) -> str | None:
    """First filename token if it is a known polymer class, else None."""
    token = re.split(r"[\s_\-.]+", stem.strip())[0].upper()
    return token if token in CLASS_TO_IDX else None


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------


def load_embedder(embedder_run: str, checkpoint: str, device: torch.device) -> SmolenCNN:
    model = SmolenCNN().to(device)
    payload = load_checkpoint(Path(RUNS_DIR) / embedder_run / checkpoint, map_location=device)
    model.load_state_dict(payload["model_state"])
    model.eval()
    return model


def build_gallery(
    model: SmolenCNN, split_mode: str, seed: int, device: torch.device
) -> tuple[np.ndarray, np.ndarray]:
    """Embed the training split: the reference gallery k-NN matches against."""
    splits = prepare_splits(seed=seed, mode=split_mode)
    df = load_parquet()
    train_ids = [s for s, sp in splits.items() if sp == "train"]
    ds = SpectrumDataset(df, train_ids)
    return embed_all(model, build_eval_loader(ds, batch_size=128), device)


@torch.no_grad()
def embed_external(model: SmolenCNN, intensities_882: np.ndarray, device: torch.device) -> np.ndarray:
    x = preprocess_spectra(intensities_882).to(device)
    return model(x).cpu().numpy()


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def plot_results_overview(
    G2: np.ndarray, X2: np.ndarray, y_gallery: np.ndarray,
    ext_true: list[int | None], ext_pred: list[int],
    y_true: list[int], y_pred: list[int], out: Path,
) -> None:
    """Single hero figure: embedding map + confusion matrix side by side.

    This is the "we have a working model" slide: the left panel shows the
    model has learned to separate polymers into clean clusters and that
    freshly measured samples land in them; the right panel quantifies it.
    """
    acc = accuracy_score(y_true, y_pred)
    fig = plt.figure(figsize=(15, 6.3))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.5, 1])
    ax_map = fig.add_subplot(gs[0])
    ax_cm = fig.add_subplot(gs[1])

    draw_embedding_map(ax_map, G2, X2, y_gallery, ext_true, ext_pred,
                       sample_label="our sample")
    ax_map.set_title("Real samples land inside the model's polymer clusters")
    draw_confusion(ax_cm, y_true, y_pred)
    ax_cm.set_title("Confusion matrix")

    fig.suptitle(
        f"Real-world validation - {acc:.0%} accuracy on {len(y_true)} freshly measured FTIR spectra",
        fontsize=14, fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out, dpi=200)
    plt.close(fig)


def plot_per_sample(rows: list[dict], out: Path) -> None:
    """Horizontal bar per sample: bar length = confidence, colour = correct."""
    rows = sorted(rows, key=lambda r: (r.get("correct") is False, -r["confidence"]))
    names = [r["file"] for r in rows]
    conf = [r["confidence"] for r in rows]
    colors = []
    for r in rows:
        if r.get("correct") is True:
            colors.append("#2e7d32")
        elif r.get("correct") is False:
            colors.append("#c62828")
        else:
            colors.append("#757575")
    fig, ax = plt.subplots(figsize=(8, max(3.0, 0.42 * len(rows))))
    ax.barh(names, conf, color=colors)
    ax.set_xlim(0, 1.0)
    ax.invert_yaxis()
    ax.set_xlabel("k-NN confidence (fraction of neighbours agreeing)")
    ax.set_title("Per-sample prediction  (green = correct, red = wrong, grey = label unknown)")
    for i, r in enumerate(rows):
        tag = f"-> {r['pred']}"
        if r.get("true"):
            tag += f"  (true {r['true']})"
        ax.text(min(r["confidence"] + 0.02, 0.98), i, tag,
                va="center", ha="left", fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Classify external FTIR .txt spectra.")
    p.add_argument("--samples", required=True, help="Folder of .txt spectra")
    p.add_argument("--embedder-run", required=True, help="Run name under runs/")
    p.add_argument("--checkpoint", default="best.pt")
    p.add_argument("--split-mode", choices=("random", "source_out", "source_cv", "source_cv2"), default="random",
                   help="Which training split supplies the k-NN gallery "
                        "(use the mode the embedder was trained with).")
    p.add_argument("--k", type=int, default=5, help="k-NN neighbours")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-plots", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = get_device()

    samples_dir = Path(args.samples)
    txt_files = sorted(samples_dir.glob("*.txt"))
    if not txt_files:
        raise FileNotFoundError(f"No .txt files in {samples_dir}")

    # --- parse + preprocess every external spectrum -----------------------
    parsed: list[dict] = []
    intensities: list[np.ndarray] = []
    for f in txt_files:
        wn, y, y_units = load_txt_spectrum(f)
        if wn.size < 50:
            print(f"  [skip] {f.name}: too few data points")
            continue
        absb = pct_T_to_absorbance(y) if "T" in y_units else y.astype(np.float32)
        grid = resample_to_canonical(wn, absb)
        if grid is None:
            print(f"  [skip] {f.name}: does not cover the {CANONICAL_LO:.0f}-{CANONICAL_HI:.0f} cm-1 range")
            continue
        parsed.append({"file": f.name, "true": label_from_filename(f.stem), "y_units": y_units})
        intensities.append(grid)

    if not parsed:
        raise RuntimeError("No usable spectra after parsing.")
    print(f"Parsed {len(parsed)}/{len(txt_files)} spectra from {samples_dir}")

    # --- embed + classify -------------------------------------------------
    model = load_embedder(args.embedder_run, args.checkpoint, device)
    E_gallery, y_gallery = build_gallery(model, args.split_mode, args.seed, device)
    E_ext = embed_external(model, np.stack(intensities), device)

    knn = KNeighborsClassifier(n_neighbors=args.k, metric="minkowski", p=2)
    knn.fit(E_gallery, y_gallery)
    proba = knn.predict_proba(E_ext)
    pred_idx = knn.classes_[proba.argmax(axis=1)]
    confidence = proba.max(axis=1)

    rows: list[dict] = []
    for rec, p_idx, conf in zip(parsed, pred_idx, confidence):
        pred_name = IDX_TO_CLASS[int(p_idx)]
        correct = None if rec["true"] is None else (pred_name == rec["true"])
        rows.append({
            "file":       rec["file"],
            "true":       rec["true"],
            "pred":       pred_name,
            "pred_idx":   int(p_idx),
            "confidence": float(conf),
            "correct":    correct,
        })

    # --- report -----------------------------------------------------------
    print(f"\n{'file':<22} {'true':<6} {'pred':<6} {'conf':>6}   result")
    print("-" * 56)
    for r in rows:
        mark = "" if r["correct"] is None else (" OK" if r["correct"] else " <-- wrong")
        print(f"{r['file']:<22} {str(r['true'] or '?'):<6} {r['pred']:<6} {r['confidence']:>6.2f}  {mark}")

    labeled = [r for r in rows if r["correct"] is not None]
    metrics: dict = {}
    if labeled:
        y_true = [CLASS_TO_IDX[r["true"]] for r in labeled]
        y_pred = [r["pred_idx"] for r in labeled]
        metrics = {
            "n_labeled":  len(labeled),
            "accuracy":   float(accuracy_score(y_true, y_pred)),
            "macro_f1":   float(f1_score(y_true, y_pred, average="macro")),
        }
        print(f"\nAccuracy   : {metrics['accuracy']:.1%}  ({len(labeled)} labelled samples)")
        print(f"Macro-F1   : {metrics['macro_f1']:.3f}")

    # --- outputs ----------------------------------------------------------
    safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", samples_dir.name).strip("_")
    out_dir = Path(RUNS_DIR) / args.embedder_run / f"predict_{safe_name}"
    out_dir.mkdir(parents=True, exist_ok=True)

    import csv
    with (out_dir / "predictions.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["file", "true", "pred", "confidence", "correct"])
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in ("file", "true", "pred", "confidence", "correct")})
    write_json(out_dir / "summary.json", {
        "samples_dir":  str(samples_dir),
        "embedder_run": args.embedder_run,
        "split_mode":   args.split_mode,
        "k":            args.k,
        "metrics":      metrics,
        "predictions":  rows,
    })

    if not args.no_plots:
        plot_per_sample(rows, out_dir / "per_sample.png")

        # 2-D embedding map: t-SNE the gallery, place samples by neighbours.
        ext_true = [CLASS_TO_IDX[r["true"]] if r["true"] else None for r in rows]
        ext_pred = [r["pred_idx"] for r in rows]
        G2, X2 = compute_projection(E_gallery, E_ext, k=args.k)
        save_embedding_map(
            G2, X2, y_gallery, ext_true, ext_pred, out_dir / "embedding_map.png",
            title="Our samples in the model's learned embedding space",
            sample_label="our sample",
        )

        if labeled:
            y_true = [CLASS_TO_IDX[r["true"]] for r in labeled]
            y_pred = [r["pred_idx"] for r in labeled]
            acc = accuracy_score(y_true, y_pred)
            save_confusion_matrix(
                y_true, y_pred, out_dir / "confusion_matrix.png",
                title=f"Confusion matrix - our samples\naccuracy = {acc:.1%}  (n = {len(y_true)})",
            )
            # Hero figure: embedding map + confusion matrix in one image.
            plot_results_overview(G2, X2, y_gallery, ext_true, ext_pred,
                                  y_true, y_pred, out_dir / "results_overview.png")

    print(f"\nResults + figures written to {out_dir}")


if __name__ == "__main__":
    main()