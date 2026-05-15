"""Embedder training loop: Smolen CNN + multi-similarity loss.

Pair-mining DataLoader, AdamW + cosine LR, multi-similarity loss +
miner, val multisim loss + KNN macro-F1 every epoch, best/latest
checkpoints, early stopping on val loss.

Run:
  python -m src.train --run-name baseline_multisim
  python -m src.train --loss triplet --run-name baseline_triplet
"""
from __future__ import annotations

import argparse
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from pytorch_metric_learning import distances, losses, miners
from sklearn.metrics import f1_score
from sklearn.neighbors import KNeighborsClassifier
from torch import nn

from src.config import DEFAULT_SEED, RUNS_DIR
from src.data import (
    SpectrumDataset,
    build_eval_loader,
    build_train_loader_pairmining,
    load_parquet,
    prepare_splits,
)
from src.model import SmolenCNN
from src.utils import (
    CSVLogger,
    CheckpointPayload,
    get_device,
    save_checkpoint,
    set_seeds,
    write_json,
)


# ---------------------------------------------------------------------------
# Loss / miner factory
# ---------------------------------------------------------------------------


def build_loss_and_miner(name: str) -> tuple[nn.Module, object]:
    """Return (loss_fn, miner) for the requested metric-learning recipe."""
    cos = distances.CosineSimilarity()
    if name == "multisim":
        loss_fn = losses.MultiSimilarityLoss(alpha=2.0, beta=50.0, base=0.5, distance=cos)
        miner = miners.MultiSimilarityMiner(epsilon=0.1, distance=cos)
    elif name == "triplet":
        loss_fn = losses.TripletMarginLoss(margin=0.1, distance=cos)
        miner = miners.TripletMarginMiner(margin=0.1, distance=cos, type_of_triplets="semihard")
    else:
        raise ValueError(f"Unknown loss recipe: {name!r}")
    return loss_fn, miner


# ---------------------------------------------------------------------------
# Per-epoch training / evaluation
# ---------------------------------------------------------------------------


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    loss_fn: nn.Module,
    miner: object,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    losses_seen: list[float] = []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        emb = model(x)
        pairs = miner(emb, y)
        loss = loss_fn(emb, y, pairs)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses_seen.append(loss.item())
    return float(np.mean(losses_seen))


@torch.no_grad()
def embed_all(model: nn.Module, loader: torch.utils.data.DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    """Return (N, D) embeddings and (N,) labels for the loader's dataset."""
    model.eval()
    chunks_e, chunks_y = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        e = model(x).cpu().numpy()
        chunks_e.append(e)
        chunks_y.append(y.numpy())
    return np.concatenate(chunks_e, axis=0), np.concatenate(chunks_y, axis=0)


@torch.no_grad()
def eval_one_epoch(
    model: nn.Module,
    val_loader: torch.utils.data.DataLoader,
    train_loader_for_ref: torch.utils.data.DataLoader,
    loss_fn: nn.Module,
    miner: object,
    device: torch.device,
) -> dict[str, float]:
    """Compute val multisim loss + KNN(k=5) macro-F1 on val embeddings."""
    model.eval()

    # Val loss using the same loss + miner.
    val_losses: list[float] = []
    for x, y in val_loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        emb = model(x)
        pairs = miner(emb, y)
        # Some miners return no pairs at all on small/easy batches; skip
        # those rather than back-propagate NaN.
        if any(p.numel() == 0 for p in pairs):
            continue
        val_losses.append(loss_fn(emb, y, pairs).item())

    # KNN macro-F1 against the (unsampled) train set: this correlates
    # with end-task accuracy better than val multisim loss alone.
    E_train, y_train = embed_all(model, train_loader_for_ref, device)
    E_val, y_val = embed_all(model, val_loader, device)
    knn = KNeighborsClassifier(n_neighbors=5, metric="minkowski", p=2)
    knn.fit(E_train, y_train)
    y_pred = knn.predict(E_val)
    macro_f1 = f1_score(y_val, y_pred, average="macro")

    return {
        "val_loss": float(np.mean(val_losses)) if val_losses else float("nan"),
        "val_knn_macro_f1": float(macro_f1),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train the Smolen-style embedder.")
    p.add_argument("--run-name", default=None, help="Subfolder under runs/. Defaults to a timestamp.")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--patience", type=int, default=20)
    # Default batch_size = m * NUM_CLASSES so MPerClassSampler returns
    # perfectly balanced batches. m * num_classes must be >= batch_size
    # for the sampler to satisfy its single-pass guarantee.
    p.add_argument("--batch-size", type=int, default=48)
    p.add_argument("--m", type=int, default=8, help="MPerClassSampler examples per class")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--loss", choices=("multisim", "triplet"), default="multisim")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--refresh-splits", action="store_true", help="Recompute split assignment")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seeds(args.seed)
    device = get_device()

    run_name = args.run_name or datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = Path(RUNS_DIR) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(run_dir / "config.json", vars(args) | {"device": str(device)})

    # Data
    splits = prepare_splits(seed=args.seed, force=args.refresh_splits)
    df = load_parquet()
    train_ids = [s for s, sp in splits.items() if sp == "train"]
    val_ids   = [s for s, sp in splits.items() if sp == "val"]
    train_ds = SpectrumDataset(df, train_ids)
    val_ds   = SpectrumDataset(df, val_ids)

    train_loader     = build_train_loader_pairmining(train_ds, batch_size=args.batch_size, m=args.m, num_workers=args.num_workers)
    train_ref_loader = build_eval_loader(train_ds, batch_size=max(args.batch_size, 128), num_workers=args.num_workers)
    val_loader       = build_eval_loader(val_ds, batch_size=max(args.batch_size, 128), num_workers=args.num_workers)

    # Model / loss / optimizer
    model = SmolenCNN().to(device)
    loss_fn, miner = build_loss_and_miner(args.loss)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    logger = CSVLogger(run_dir / "metrics.csv")
    best_val_loss = float("inf")
    best_epoch = 0
    epochs_since_best = 0

    print(f"[run {run_name}] device={device} | train={len(train_ds)} val={len(val_ds)} | loss={args.loss}")
    if device.type == "cuda":
        # Confirm the model + a sample batch actually land on the GPU.
        _x, _y = next(iter(train_loader))
        _ = model(_x.to(device))
        torch.cuda.synchronize()
        mem_mb = torch.cuda.memory_allocated(device) / 1024 / 1024
        print(f"[run {run_name}] cuda device: {torch.cuda.get_device_name(device)} | allocated after warmup: {mem_mb:.1f} MB")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, loss_fn, miner, optimizer, device)
        scheduler.step()
        metrics = eval_one_epoch(model, val_loader, train_ref_loader, loss_fn, miner, device)
        elapsed = time.time() - t0

        improved = metrics["val_loss"] < best_val_loss - 1e-4
        if improved:
            best_val_loss = metrics["val_loss"]
            best_epoch = epoch
            epochs_since_best = 0
            save_checkpoint(
                run_dir / "best.pt",
                CheckpointPayload(
                    epoch=epoch,
                    model_state=model.state_dict(),
                    optimizer_state=optimizer.state_dict(),
                    scheduler_state=scheduler.state_dict(),
                    metric_value=best_val_loss,
                    metric_name="val_loss",
                    seed=args.seed,
                ),
            )
        else:
            epochs_since_best += 1

        save_checkpoint(
            run_dir / "latest.pt",
            CheckpointPayload(
                epoch=epoch,
                model_state=model.state_dict(),
                optimizer_state=optimizer.state_dict(),
                scheduler_state=scheduler.state_dict(),
                metric_value=metrics["val_loss"],
                metric_name="val_loss",
                seed=args.seed,
            ),
        )

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": metrics["val_loss"],
            "val_knn_macro_f1": metrics["val_knn_macro_f1"],
            "lr": optimizer.param_groups[0]["lr"],
            "elapsed_s": elapsed,
            "best_epoch": best_epoch,
        }
        logger.log(row)
        flag = "*" if improved else " "
        print(f"  ep {epoch:3d} {flag} train={train_loss:.4f} val={metrics['val_loss']:.4f} f1={metrics['val_knn_macro_f1']:.4f} lr={row['lr']:.2e} ({elapsed:.1f}s)")

        if epochs_since_best >= args.patience:
            print(f"  early stop at epoch {epoch} (no improvement for {args.patience} epochs)")
            break

    print(f"[run {run_name}] best val_loss={best_val_loss:.4f} @ epoch {best_epoch}")
    print(f"[run {run_name}] checkpoints in {run_dir}")


if __name__ == "__main__":
    main()
