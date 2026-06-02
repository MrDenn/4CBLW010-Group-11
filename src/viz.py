"""Shared plotting helpers for predict.py and evaluate.py.

The model classifies by k-NN in embedding space, so every figure here
treats classification the same way: `compute_projection` lays out the
training gallery with t-SNE and places each evaluated spectrum among the
gallery neighbours that voted on it, and all confusion matrices share one
visual style so they read cleanly side by side in a slide deck.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: render straight to PNG files

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.lines import Line2D
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.neighbors import NearestNeighbors

from src.config import IDX_TO_CLASS, POLYMER_CLASSES

OK_EDGE = "#111111"
BAD_EDGE = "#c62828"


# ---------------------------------------------------------------------------
# Axis-level drawing primitives (render into an existing axis)
# ---------------------------------------------------------------------------


def draw_confusion(ax: plt.Axes, y_true: list[int], y_pred: list[int]) -> None:
    """Render a confusion-matrix heatmap into an existing axis."""
    classes = sorted(set(y_true) | set(y_pred))
    names = [IDX_TO_CLASS[c] for c in classes]
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=names, yticklabels=names, ax=ax,
                annot_kws={"size": 13})
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")


def draw_embedding_map(
    ax: plt.Axes,
    G2: np.ndarray, X2: np.ndarray,
    y_gallery: np.ndarray,
    ext_true: list[int | None], ext_pred: list[int],
    sample_label: str = "sample",
) -> None:
    """Render the 2-D embedding scatter into an existing axis.

    Training gallery = small faded dots coloured by true class. Evaluated
    spectra = large circles coloured by their *true* class, with a black
    outline if correctly classified and a red outline if not - so a circle
    sitting in a cloud of a different colour is a visible misclassification.
    """
    palette = sns.color_palette("tab10", n_colors=len(POLYMER_CLASSES))

    for idx in IDX_TO_CLASS:
        m = y_gallery == idx
        if m.any():
            ax.scatter(G2[m, 0], G2[m, 1], s=10, alpha=0.16,
                       color=palette[idx], linewidths=0)

    for i in range(len(X2)):
        t, p = ext_true[i], ext_pred[i]
        color = palette[t] if t is not None else palette[p]
        edge = OK_EDGE if (t is None or t == p) else BAD_EDGE
        ax.scatter(X2[i, 0], X2[i, 1], s=94, marker="o",
                   color=color, edgecolor=edge, linewidth=1.7, zorder=5)

    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")

    handles = [
        Line2D([], [], marker="o", linestyle="", markersize=8,
               color=palette[i], alpha=0.5, label=cls)
        for i, cls in IDX_TO_CLASS.items()
    ]
    handles += [
        Line2D([], [], marker="o", linestyle="", markersize=8, color="#bdbdbd",
               markeredgecolor=OK_EDGE, markeredgewidth=1.7, label=f"{sample_label} (correct)"),
        Line2D([], [], marker="o", linestyle="", markersize=8, color="#bdbdbd",
               markeredgecolor=BAD_EDGE, markeredgewidth=1.7, label=f"{sample_label} (misclassified)"),
    ]
    ax.legend(handles=handles, fontsize=7, loc="best", framealpha=0.9, ncol=2)


# ---------------------------------------------------------------------------
# Projection
# ---------------------------------------------------------------------------


def compute_projection(
    E_gallery: np.ndarray, E_ext: np.ndarray, k: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """t-SNE the gallery, then place each external sample at the mean 2-D
    position of its k nearest gallery neighbours.

    t-SNE has no `.transform()` and tends to isolate small groups of
    near-identical points into their own island if they are included in
    the fit. Instead we fit t-SNE on the gallery alone, then locate each
    external sample by the *same* k neighbours the k-NN classifier votes
    over - so a circle physically sits among the points that decided its
    class. A small jitter keeps near-identical samples individually
    visible. sklearn's t-SNE has no numba dependency.
    """
    perplexity = float(min(30, max(5, (len(E_gallery) - 1) // 3)))
    G2 = TSNE(n_components=2, perplexity=perplexity, init="pca",
              random_state=42).fit_transform(E_gallery)

    nn = NearestNeighbors(n_neighbors=k).fit(E_gallery)
    _, idx = nn.kneighbors(E_ext)
    X2 = G2[idx].mean(axis=1)

    span = G2.max(axis=0) - G2.min(axis=0)
    jitter = np.random.default_rng(42).normal(0.0, 0.015, X2.shape) * span
    return G2, X2 + jitter


# ---------------------------------------------------------------------------
# Figure-level helpers (own figure, save to disk)
# ---------------------------------------------------------------------------


def save_confusion_matrix(
    y_true: list[int], y_pred: list[int], out: Path, title: str | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 4.6))
    draw_confusion(ax, y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    ax.set_title(title or f"Confusion matrix  (accuracy = {acc:.1%}, n = {len(y_true)})")
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def save_embedding_map(
    G2: np.ndarray, X2: np.ndarray, y_gallery: np.ndarray,
    ext_true: list[int | None], ext_pred: list[int], out: Path,
    title: str | None = None, sample_label: str = "sample",
) -> None:
    fig, ax = plt.subplots(figsize=(7.8, 6.2))
    draw_embedding_map(ax, G2, X2, y_gallery, ext_true, ext_pred, sample_label)
    ax.set_title(title or "Embedding map")
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def save_confusion_grid(
    panels: list[tuple[str, list[int], list[int]]], out: Path,
    suptitle: str | None = None,
) -> None:
    """Render several confusion matrices in a row - one per (title, y_true,
    y_pred) panel - so different test sets sit side by side in one image."""
    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(5.2 * n, 4.7), squeeze=False)
    for ax, (title, y_true, y_pred) in zip(axes[0], panels):
        draw_confusion(ax, y_true, y_pred)
        acc = accuracy_score(y_true, y_pred)
        ax.set_title(f"{title}\naccuracy = {acc:.1%}  (n = {len(y_true)})")
    if suptitle:
        fig.suptitle(suptitle, fontsize=14, fontweight="bold")
        fig.tight_layout(rect=(0, 0, 1, 0.92))
    else:
        fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)