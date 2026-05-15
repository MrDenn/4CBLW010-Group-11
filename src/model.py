"""Smolen Fig. 1A architecture: 4 conv blocks + 3 dense layers.

Two heads are provided:
  - SmolenCNN: 24-dim L2-normalized embedding for metric learning.
  - SmolenCNNClassifier: Linear(120, 6) with ReLU activation for the
    cross-entropy baseline.

The trunk dimensions are dictated by Smolen Fig. 1A (valid-padded
convolutions, max-pool stride 2, flatten of 52*64 = 3328).
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from src.config import (
    CONV_CHANNELS,
    CONV_KERNEL,
    DENSE_DIMS,
    EMBED_DIM,
    INPUT_LEN_PADDED,
    NUM_CLASSES,
)


class _SmolenTrunk(nn.Module):
    """Shared feature extractor: 4 (Conv1d -> BN -> ReLU -> MaxPool) + 2 dense.

    Returns the 120-dim activation that immediately precedes the head.
    BatchNorm1d after each conv is the addition that the raw Fig. 1A
    diagram omits but the reference TF implementation includes.
    """

    def __init__(self) -> None:
        super().__init__()
        blocks: list[nn.Module] = []
        for in_ch, out_ch in zip(CONV_CHANNELS[:-1], CONV_CHANNELS[1:]):
            blocks.append(nn.Conv1d(in_ch, out_ch, kernel_size=CONV_KERNEL))
            blocks.append(nn.BatchNorm1d(out_ch))
            blocks.append(nn.ReLU(inplace=True))
            blocks.append(nn.MaxPool1d(kernel_size=2))
        self.conv = nn.Sequential(*blocks)
        self.flatten = nn.Flatten()
        flat_dim, fc1_dim, fc2_dim = DENSE_DIMS
        self.fc1 = nn.Linear(flat_dim, fc1_dim)
        self.fc2 = nn.Linear(fc1_dim, fc2_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv(x)
        h = self.flatten(h)
        h = F.relu(self.fc1(h), inplace=True)
        h = F.relu(self.fc2(h), inplace=True)
        return h


class SmolenCNN(nn.Module):
    """Embedding head: trunk -> Linear(120, 24) -> L2 normalize.

    The L2 normalization places embeddings on the unit hypersphere where
    cosine similarity equals the dot product, which is what
    `MultiSimilarityLoss` with `CosineSimilarity()` assumes.
    """

    def __init__(self, embed_dim: int = EMBED_DIM) -> None:
        super().__init__()
        self.trunk = _SmolenTrunk()
        self.embed = nn.Linear(DENSE_DIMS[-1], embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.trunk(x)
        e = self.embed(h)
        return F.normalize(e, p=2, dim=1)


class SmolenCNNClassifier(nn.Module):
    """Classification baseline: trunk -> Linear(120, 6) -> ReLU.

    The ReLU on logits is Smolen's verbatim spec; PyTorch's CrossEntropyLoss
    will still work because the loss takes the softmax-CE of whatever it
    receives. If training is pathological, drop the ReLU.
    """

    def __init__(self, num_classes: int = NUM_CLASSES) -> None:
        super().__init__()
        self.trunk = _SmolenTrunk()
        self.head = nn.Linear(DENSE_DIMS[-1], num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.trunk(x)
        logits = self.head(h)
        return F.relu(logits, inplace=True)


def smoke_test() -> None:
    """Manual sanity check called from `python -m src.model`."""
    x = torch.randn(8, 1, INPUT_LEN_PADDED)
    emb_model = SmolenCNN()
    emb = emb_model(x)
    assert emb.shape == (8, EMBED_DIM), emb.shape
    norms = emb.norm(p=2, dim=1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5), norms
    print(f"SmolenCNN OK: out {tuple(emb.shape)}, norms ~1.0")

    clf = SmolenCNNClassifier()
    logits = clf(x)
    assert logits.shape == (8, NUM_CLASSES), logits.shape
    n_params = sum(p.numel() for p in emb_model.parameters())
    print(f"SmolenCNNClassifier OK: out {tuple(logits.shape)} | param count {n_params:,}")


if __name__ == "__main__":
    smoke_test()
