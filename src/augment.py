"""On-the-fly physics-based augmentation for ATR-FTIR spectra.

Applied to the TRAINING split only, in absorbance space, on the 882-point
canonical grid, BEFORE min/max normalization and zero-padding (which live
in `data.preprocess_spectra`). The goal is to synthesize the
instrument/measurement nuisance that the pristine Villegas bulk lacks, so
the embedder learns band identity rather than instrument fingerprint.

Critical constraint (see research-notes/augmentation_implementation_report.md
§2): per-spectrum min/max normalization downstream silently cancels any
pure constant offset and any pure global multiplicative scale. Every
transform here is therefore wavenumber-dependent (slopes/ramps/curvature)
or per-channel (noise/shift/broaden) so that it survives normalization by
changing the spectrum's *shape*.

Five transforms, applied in this order:
  4.1 smooth multiplicative field  (ATR penetration-depth tilt + scatter)  PE-safe
  4.2 smooth additive baseline     (drift / fouling / scatter)             PE-safe
  4.3 additive Gaussian noise      (detector / shot noise)                 PE-safe
  4.4 wavenumber shift             (calibration differences)               PE-RISKY (mild)
  4.5 peak broadening              (resolution / apodization)              PE-RISKY (mild)

Transforms 4.1-4.3 change only band heights/baselines and never move peak
positions, so they are safe to apply generously. Transforms 4.4-4.5
perturb the wavenumber axis and can erase the narrow features that separate
HDPE from LDPE, so they are kept mild and gated.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter1d

from src.config import AUG, INPUT_LEN_RAW

# Normalized wavenumber axis u[k] = (k - (N-1)/2) / N in [-0.5, 0.5].
_U = ((np.arange(INPUT_LEN_RAW) - (INPUT_LEN_RAW - 1) / 2.0) / INPUT_LEN_RAW).astype(np.float32)
# Centered quadratic basis (mean ~0 over the axis) so the curvature term
# carries no net constant component.
_U2 = (_U ** 2 - 1.0 / 12.0).astype(np.float32)


class Augmenter:
    """Composable per-spectrum spectroscopic augmenter.

    Parameters
    ----------
    cfg : dict
        An AUG-style dict (see src.config.AUG). Each transform reads its own
        sub-dict; magnitudes are sampled per call.
    rng : np.random.Generator
        Source of randomness. Use `reseed` to give DataLoader workers
        independent streams.
    """

    def __init__(self, cfg: dict | None = None, rng: np.random.Generator | None = None) -> None:
        self.cfg = AUG if cfg is None else cfg
        self.rng = np.random.default_rng() if rng is None else rng
        self._epoch = 0

    # -- lifecycle ---------------------------------------------------------

    def reseed(self, seed: int) -> None:
        self.rng = np.random.default_rng(seed)

    def set_epoch(self, epoch: int) -> None:
        """Used only for the optional magnitude curriculum warm-up."""
        self._epoch = int(epoch)

    def _curriculum_scale(self) -> float:
        warm = int(self.cfg.get("curriculum_warmup_epochs", 0) or 0)
        if warm <= 0:
            return 1.0
        return float(min(1.0, (self._epoch + 1) / warm))

    # -- individual transforms (each takes/returns an 882 float32 vector) --

    def _mult_field(self, a: np.ndarray, scale: float) -> np.ndarray:
        """4.1 Smooth multiplicative field: ATR penetration-depth tilt + scatter."""
        p = self.cfg["mult_field"]
        c1 = self.rng.uniform(*p["c1"]) * scale
        c2 = self.rng.uniform(*p["c2"]) * scale
        f = 1.0 + c1 * _U + c2 * _U2
        f = np.clip(f, p["clip"][0], p["clip"][1])
        return a * f

    def _add_baseline(self, a: np.ndarray, rng_amp: float, scale: float) -> np.ndarray:
        """4.2 Smooth additive sloped/curved baseline, amplitude relative to range."""
        p = self.cfg["add_baseline"]
        g1, g2, g0 = self.rng.uniform(-1.0, 1.0, size=3)
        shape = g1 * _U + g2 * _U2 + g0 * 0.5  # g0 kept small; pure-constant part is cancelled by norm
        ptp = np.ptp(shape)
        if ptp > 1e-8:
            shape = shape / ptp  # peak-to-peak ~1
        beta = self.rng.uniform(*p["beta"]) * scale
        return a + beta * rng_amp * shape.astype(np.float32)

    def _noise(self, a: np.ndarray, ref: float, scale: float) -> np.ndarray:
        """4.3 Additive white Gaussian noise.

        Sigma is a fraction of the spectrum's absorbance range. The report
        specifies "fraction of max absorbance", but absorbance can be
        negative here (we keep above-baseline noise as negative A), so the
        range is the robust, always-non-negative amplitude reference.
        """
        rho = self.rng.uniform(*self.cfg["noise"]["rho"]) * scale
        sigma = max(rho * ref, 0.0)
        if sigma < 1e-12:
            return a
        return a + self.rng.normal(0.0, sigma, size=a.shape).astype(np.float32)

    def _wn_shift(self, a: np.ndarray, scale: float) -> np.ndarray:
        """4.4 Sub-pixel wavenumber shift via interpolation, EDGE-filled."""
        lo, hi = self.cfg["wn_shift"]["channels"]
        delta = self.rng.uniform(lo, hi) * scale
        if abs(delta) < 1e-6:
            return a
        idx = np.arange(INPUT_LEN_RAW, dtype=np.float32)
        # Sample the original spectrum at shifted positions; np.interp clamps
        # out-of-range queries to edge values (edge fill, not zero fill).
        return np.interp(idx + delta, idx, a).astype(np.float32)

    def _broaden(self, a: np.ndarray, scale: float) -> np.ndarray:
        """4.5 Gaussian peak broadening (resolution/apodization differences)."""
        lo, hi = self.cfg["broaden"]["sigma_ch"]
        sigma = self.rng.uniform(lo, hi) * scale
        if sigma < 1e-6:
            return a
        return gaussian_filter1d(a, sigma=sigma, mode="nearest").astype(np.float32)

    # -- composition -------------------------------------------------------

    def __call__(self, a: np.ndarray) -> np.ndarray:
        """Augment one 882-point absorbance vector. Returns a new vector."""
        a = np.asarray(a, dtype=np.float32)
        if a.shape != (INPUT_LEN_RAW,):
            raise ValueError(f"Augmenter expects ({INPUT_LEN_RAW},), got {a.shape}")
        if not self.cfg.get("enabled", True):
            return a

        scale = self._curriculum_scale()
        p_each = float(self.cfg.get("p_each", 0.7))
        rng_amp = float(np.ptp(a))          # this spectrum's pre-norm range R (>= 0)

        out = a.copy()
        # Height / baseline / noise first (safe), then axis (risky).
        if self.rng.random() < p_each:
            out = self._mult_field(out, scale)
        if self.rng.random() < p_each:
            out = self._add_baseline(out, rng_amp, scale)
        if self.rng.random() < p_each:
            out = self._noise(out, rng_amp, scale)
        if self.rng.random() < p_each:
            out = self._wn_shift(out, scale)
        if self.rng.random() < p_each:
            out = self._broaden(out, scale)
        return out


def smoke_test() -> None:
    """Manual checks called from `python -m src.augment`.

    Mirrors research-notes/augmentation_implementation_report.md §6:
      (a) output length is 882;
      (b) a constant-offset-only input normalizes identically (proves the
          min/max-cancellation understanding of §2);
      (c) with all magnitude ranges set to 0, __call__ is the identity.
    """
    rng = np.random.default_rng(0)
    base = np.cumsum(rng.normal(size=INPUT_LEN_RAW)).astype(np.float32)  # smooth-ish signal

    # (a) shape preserved
    aug = Augmenter(rng=np.random.default_rng(1))
    out = aug(base)
    assert out.shape == (INPUT_LEN_RAW,), out.shape

    # (b) a pure constant offset washes out under downstream min/max norm.
    def norm(x: np.ndarray) -> np.ndarray:
        return (x - x.min()) / (x.max() - x.min() + 1e-8)
    assert np.allclose(norm(base), norm(base + 5.0), atol=1e-6), "constant offset should cancel under norm"

    # (c) zeroed ranges -> identity
    zero_cfg = {
        "enabled": True, "p_each": 1.0,
        "mult_field":  {"c1": (0.0, 0.0), "c2": (0.0, 0.0), "clip": (0.5, 1.8)},
        "add_baseline":{"beta": (0.0, 0.0), "order": 2},
        "noise":       {"rho": (0.0, 0.0)},
        "wn_shift":    {"channels": (0.0, 0.0)},
        "broaden":     {"sigma_ch": (0.0, 0.0)},
        "curriculum_warmup_epochs": 0,
    }
    ident = Augmenter(cfg=zero_cfg, rng=np.random.default_rng(2))
    assert np.allclose(ident(base), base, atol=1e-6), "zeroed augmenter must be identity"

    # (d) default augmenter actually changes the shape (sanity that it bites).
    changed = not np.allclose(norm(out), norm(base), atol=1e-3)
    print(f"Augmenter OK: out {out.shape}, identity+offset checks pass, "
          f"default augmentation alters shape={changed}")


if __name__ == "__main__":
    smoke_test()
