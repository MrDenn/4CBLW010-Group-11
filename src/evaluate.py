"""Downstream classifier comparison.

Reproduces Smolen Fig. 2A:
  - 5 sklearn classifiers (KNN, SVC-RBF, SVC-poly, LDA, QDA, PLS-DA)
    on the trained embedder's 24-dim embeddings.
  - The same 5 classifiers on the raw min/max-normalized spectra
    (no embedding) as a baseline.
  - Optional: the SmolenCNNClassifier head trained with CE.

Usage:
  python -m src.evaluate --embedder-run baseline_multisim
  python -m src.evaluate --embedder-run baseline_multisim --classifier-run baseline_ce
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from sklearn.cross_decomposition import PLSRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from src.config import IDX_TO_CLASS, NUM_CLASSES, RUNS_DIR
from src.data import (
    SpectrumDataset,
    build_eval_loader,
    load_parquet,
    prepare_splits,
)
from src.model import SmolenCNN, SmolenCNNClassifier
from src.train import embed_all
from src.utils import get_device, load_checkpoint, write_json
from src.viz import (
    compute_projection,
    save_confusion_grid,
    save_confusion_matrix,
    save_embedding_map,
)


# ---------------------------------------------------------------------------
# Classifier zoo (Smolen §9 verbatim settings)
# ---------------------------------------------------------------------------


def build_sklearn_zoo(seed: int) -> dict[str, object]:
    """The classifiers Smolen Fig. 2A compares; settings carried over verbatim."""
    return {
        "KNN(k=5)":  KNeighborsClassifier(n_neighbors=5, metric="minkowski", p=2, weights="uniform"),
        "SVC-RBF":   SVC(kernel="rbf",  C=1.0, gamma="scale", probability=True, random_state=seed),
        "SVC-poly3": SVC(kernel="poly", degree=3, C=1.0, probability=True, random_state=seed),
        "LDA":       LinearDiscriminantAnalysis(),
        "QDA":       QuadraticDiscriminantAnalysis(),
    }


def fit_plsda(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray) -> np.ndarray:
    """PLS-DA: regress one-hot labels with PLS, argmax the prediction.

    Components = num_classes - 1 to match Smolen's setting.
    """
    Y_one_hot = np.eye(NUM_CLASSES)[y_train]
    pls = PLSRegression(n_components=NUM_CLASSES - 1)
    pls.fit(X_train, Y_one_hot)
    return pls.predict(X_test).argmax(axis=1)


# ---------------------------------------------------------------------------
# Raw vs. embedded feature matrices
# ---------------------------------------------------------------------------


def raw_features(ds: SpectrumDataset) -> tuple[np.ndarray, np.ndarray]:
    """Flatten the dataset's preprocessed (normalized + padded) tensor.

    Shape (N, 1, 896) -> (N, 896). This is the "no-embedding" baseline:
    the same sklearn classifiers applied directly to the spectra.
    """
    X = ds.X.numpy().reshape(len(ds), -1)
    y = ds.y.numpy()
    return X, y


def embedded_features(
    ckpt_path: Path,
    datasets: dict[str, SpectrumDataset],
    device: torch.device,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    model = SmolenCNN().to(device)
    payload = load_checkpoint(ckpt_path, map_location=device)
    model.load_state_dict(payload["model_state"])
    model.eval()
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for split_name, ds in datasets.items():
        loader = build_eval_loader(ds, batch_size=128)
        out[split_name] = embed_all(model, loader, device)
    return out


# ---------------------------------------------------------------------------
# Comparison driver
# ---------------------------------------------------------------------------


def score_zoo(
    features: dict[str, tuple[np.ndarray, np.ndarray]],
    label: str,
    seed: int,
) -> list[dict[str, float | str]]:
    X_train, y_train = features["train"]
    X_test,  y_test  = features["test"]
    rows: list[dict[str, float | str]] = []

    zoo = build_sklearn_zoo(seed)
    for name, clf in zoo.items():
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        rows.append({
            "feature_space": label,
            "classifier":    name,
            "test_acc":      float(accuracy_score(y_test, y_pred)),
            "test_macro_f1": float(f1_score(y_test, y_pred, average="macro")),
        })

    y_pred = fit_plsda(X_train, y_train, X_test)
    rows.append({
        "feature_space": label,
        "classifier":    "PLS-DA",
        "test_acc":      float(accuracy_score(y_test, y_pred)),
        "test_macro_f1": float(f1_score(y_test, y_pred, average="macro")),
    })
    return rows


@torch.no_grad()
def score_classifier_head(ckpt_path: Path, ds_test: SpectrumDataset, device: torch.device) -> dict[str, float | str]:
    model = SmolenCNNClassifier().to(device)
    payload = load_checkpoint(ckpt_path, map_location=device)
    model.load_state_dict(payload["model_state"])
    model.eval()

    preds, ys = [], []
    for x, y in build_eval_loader(ds_test, batch_size=128):
        x = x.to(device)
        preds.append(model(x).argmax(dim=1).cpu().numpy())
        ys.append(y.numpy())
    y_pred = np.concatenate(preds)
    y_true = np.concatenate(ys)
    return {
        "feature_space": "raw (CNN classifier)",
        "classifier":    "SmolenCNNClassifier",
        "test_acc":      float(accuracy_score(y_true, y_pred)),
        "test_macro_f1": float(f1_score(y_true, y_pred, average="macro")),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Smolen Fig. 2A reproduction.")
    p.add_argument("--embedder-run", required=True, help="Run name under runs/ for the embedder")
    p.add_argument("--classifier-run", default=None, help="Optional run name for the CE classifier baseline")
    p.add_argument("--checkpoint", default="best.pt", help="Filename inside the run dir (default best.pt)")
    p.add_argument("--split-mode", choices=("random", "source_out", "source_cv"), default="random",
                   help="Must match the split mode the embedder was trained with.")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def _test_split_names(splits: dict[str, str]) -> list[str]:
    """All split names that begin with `test` (random -> ['test'];
    source_out -> ['test_FLOPP-e', 'test_OpenSpecy'])."""
    return sorted({sp for sp in splits.values() if sp.startswith("test")})


def main() -> None:
    args = parse_args()
    device = get_device()

    splits = prepare_splits(seed=args.seed, mode=args.split_mode)
    df = load_parquet()
    test_splits = _test_split_names(splits)
    if not test_splits:
        raise RuntimeError(f"No test splits found in mode={args.split_mode!r}. Splits seen: {set(splits.values())}")

    ds: dict[str, SpectrumDataset] = {
        "train": SpectrumDataset(df, [s for s, sp in splits.items() if sp == "train"]),
    }
    for tsp in test_splits:
        ds[tsp] = SpectrumDataset(df, [s for s, sp in splits.items() if sp == tsp])

    # Embedder feature space (computed once for all test splits).
    ckpt = Path(RUNS_DIR) / args.embedder_run / args.checkpoint
    emb = embedded_features(ckpt, ds, device)
    raw = {split: raw_features(ds[split]) for split in ds}

    rows: list[dict[str, float | str]] = []
    for tsp in test_splits:
        # Raw-spectrum baseline on this test split.
        raw_pair = {"train": raw["train"], "test": raw[tsp]}
        for r in score_zoo(raw_pair, label=f"raw spectra ({tsp})", seed=args.seed):
            rows.append(r)
        # Embedder feature space on this test split.
        emb_pair = {"train": emb["train"], "test": emb[tsp]}
        for r in score_zoo(emb_pair, label=f"SLE-MultiSim embeddings ({tsp})", seed=args.seed):
            rows.append(r)
        # Classifier head, if provided.
        if args.classifier_run is not None:
            clf_ckpt = Path(RUNS_DIR) / args.classifier_run / args.checkpoint
            r = score_classifier_head(clf_ckpt, ds[tsp], device)
            r["feature_space"] = f"raw (CNN classifier, {tsp})"
            rows.append(r)

    print(f"\n{'feature_space':<46} {'classifier':<22} {'acc':>6}   {'macro_f1':>8}")
    print("-" * 88)
    for r in rows:
        print(f"{r['feature_space']:<46} {r['classifier']:<22} {r['test_acc']:>6.4f}   {r['test_macro_f1']:>8.4f}")
    print()
    print(f"Class index map: {dict(IDX_TO_CLASS)}")

    out_dir = Path(RUNS_DIR) / args.embedder_run
    out_path = out_dir / f"evaluation_{args.split_mode}.json"
    write_json(out_path, {
        "rows":           rows,
        "embedder_run":   args.embedder_run,
        "classifier_run": args.classifier_run,
        "split_mode":     args.split_mode,
        "test_splits":    test_splits,
    })
    print(f"Results written to {out_path}")

    # --- figures: confusion matrix + embedding map per test split ---------
    # Predictions for the figures come from k-NN (k=5) on the embeddings,
    # i.e. the canonical Smolen inference path - the same mechanism
    # predict.py uses, so these plots are directly comparable to the
    # real-world-samples figures.
    E_train, y_train = emb["train"]
    knn = KNeighborsClassifier(n_neighbors=5, metric="minkowski", p=2)
    knn.fit(E_train, y_train)

    fig_dir = out_dir / f"figures_{args.split_mode}"
    fig_dir.mkdir(parents=True, exist_ok=True)
    panels: list[tuple[str, list[int], list[int]]] = []
    for tsp in test_splits:
        E_t, y_t = emb[tsp]
        y_pred_t = knn.predict(E_t)
        nice = tsp.replace("test_", "")

        save_confusion_matrix(
            list(y_t), list(y_pred_t), fig_dir / f"confusion_{nice}.png",
            title=f"Confusion matrix - {nice}\n"
                  f"accuracy = {accuracy_score(y_t, y_pred_t):.1%}  (n = {len(y_t)})",
        )
        G2, X2 = compute_projection(E_train, E_t, k=5)
        save_embedding_map(
            G2, X2, y_train, list(y_t), list(y_pred_t),
            fig_dir / f"embedding_map_{nice}.png",
            title=f"{nice} test set in the model's embedding space",
            sample_label=f"{nice} spectrum",
        )
        panels.append((nice, list(y_t), list(y_pred_t)))

    # Combined view: all held-out test sources pooled into one confusion
    # matrix and one embedding map. Only meaningful with >1 test source.
    if len(test_splits) > 1:
        E_all = np.concatenate([emb[t][0] for t in test_splits])
        y_all = np.concatenate([emb[t][1] for t in test_splits])
        y_pred_all = knn.predict(E_all)

        save_confusion_matrix(
            list(y_all), list(y_pred_all), fig_dir / "confusion_combined.png",
            title=f"Confusion matrix - all held-out test sources\n"
                  f"accuracy = {accuracy_score(y_all, y_pred_all):.1%}  (n = {len(y_all)})",
        )
        G2c, X2c = compute_projection(E_train, E_all, k=5)
        save_embedding_map(
            G2c, X2c, y_train, list(y_all), list(y_pred_all),
            fig_dir / "embedding_map_combined.png",
            title="All held-out test sources in the model's embedding space",
            sample_label="test spectrum",
        )
        panels.append(("all test sources", list(y_all), list(y_pred_all)))

    save_confusion_grid(
        panels, fig_dir / "confusion_matrices.png",
        suptitle="Held-out test sets - k-NN on SLE-MultiSim embeddings",
    )
    print(f"Figures written to {fig_dir}")


if __name__ == "__main__":
    main()
