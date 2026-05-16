"""Cross-entropy baseline: SmolenCNNClassifier trained with CE loss.

The comparator to the similarity-learning embedder. Same trunk, but
the head is Linear(120, 6) with ReLU on the logits, and the loss is
sparse categorical cross-entropy (PyTorch's nn.CrossEntropyLoss on
integer-class targets is the equivalent).

Run:
  python -m src.train_classifier --run-name baseline_ce
"""
from __future__ import annotations

import argparse
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch import nn

from src.config import DEFAULT_SEED, RUNS_DIR
from src.data import (
    SpectrumDataset,
    build_eval_loader,
    build_shuffled_loader,
    load_parquet,
    prepare_splits,
)
from src.model import SmolenCNNClassifier
from src.utils import (
    CSVLogger,
    CheckpointPayload,
    get_device,
    save_checkpoint,
    set_seeds,
    write_json,
)


def train_one_epoch(model, loader, loss_fn, optimizer, device) -> float:
    model.train()
    losses_seen: list[float] = []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        loss = loss_fn(logits, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses_seen.append(loss.item())
    return float(np.mean(losses_seen))


@torch.no_grad()
def evaluate(model, loader, loss_fn, device) -> dict[str, float]:
    model.eval()
    losses_seen: list[float] = []
    all_pred, all_y = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        loss = loss_fn(logits, y)
        losses_seen.append(loss.item())
        all_pred.append(logits.argmax(dim=1).cpu().numpy())
        all_y.append(y.cpu().numpy())
    y_pred = np.concatenate(all_pred)
    y_true = np.concatenate(all_y)
    return {
        "val_loss":    float(np.mean(losses_seen)),
        "val_acc":     float(accuracy_score(y_true, y_pred)),
        "val_macro_f1": float(f1_score(y_true, y_pred, average="macro")),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train the CE classification baseline.")
    p.add_argument("--run-name", default=None)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--split-mode", choices=("random", "source_out"), default="random")
    p.add_argument("--refresh-splits", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seeds(args.seed)
    device = get_device()

    run_name = args.run_name or ("ce_" + datetime.now().strftime("%Y%m%d-%H%M%S"))
    run_dir = Path(RUNS_DIR) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(run_dir / "config.json", vars(args) | {"device": str(device), "model": "SmolenCNNClassifier"})

    splits = prepare_splits(seed=args.seed, mode=args.split_mode, force=args.refresh_splits)
    df = load_parquet()
    train_ds = SpectrumDataset(df, [s for s, sp in splits.items() if sp == "train"])
    val_ds   = SpectrumDataset(df, [s for s, sp in splits.items() if sp == "val"])

    train_loader = build_shuffled_loader(train_ds, batch_size=args.batch_size, num_workers=args.num_workers)
    val_loader   = build_eval_loader(val_ds, batch_size=max(args.batch_size, 128), num_workers=args.num_workers)

    model = SmolenCNNClassifier().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    logger = CSVLogger(run_dir / "metrics.csv")
    best_val_loss = float("inf")
    best_epoch = 0
    epochs_since_best = 0

    print(f"[run {run_name}] device={device} | train={len(train_ds)} val={len(val_ds)}")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
        scheduler.step()
        metrics = evaluate(model, val_loader, loss_fn, device)
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

        row = {"epoch": epoch, "train_loss": train_loss, **metrics, "lr": optimizer.param_groups[0]["lr"], "elapsed_s": elapsed, "best_epoch": best_epoch}
        logger.log(row)
        flag = "*" if improved else " "
        print(f"  ep {epoch:3d} {flag} train={train_loss:.4f} val={metrics['val_loss']:.4f} acc={metrics['val_acc']:.4f} f1={metrics['val_macro_f1']:.4f} ({elapsed:.1f}s)")

        if epochs_since_best >= args.patience:
            print(f"  early stop at epoch {epoch}")
            break

    print(f"[run {run_name}] best val_loss={best_val_loss:.4f} @ epoch {best_epoch}")


if __name__ == "__main__":
    main()
