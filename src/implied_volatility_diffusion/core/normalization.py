"""Surface normalization helpers for implied-volatility grids."""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

DEFAULT_SIGMA_FLOOR = 1e-6
DEFAULT_IV_FLOOR = 1e-8


def _as_3d(x):
    x = np.asarray(x, float)
    if x.ndim == 2:
        x = x[None]
    if x.ndim != 3:
        raise ValueError(f"Expected (I,J) or (N,I,J), got {x.shape}")
    return x


def iv_to_log_iv(iv, floor=DEFAULT_IV_FLOOR):
    """Convert IV values to log-IV, clipping finite entries to ``floor``."""
    iv = np.asarray(iv, float)
    out = np.full_like(iv, np.nan)
    m = np.isfinite(iv)
    out[m] = np.log(np.maximum(iv[m], floor))
    return out


def log_iv_to_iv(x):
    """Convert log-IV values back to IV values via exponentiation."""
    return np.exp(np.asarray(x, float))


@dataclass
class SurfaceNormalizer:
    """Running estimator for per-grid-cell mean and std in log-IV space."""

    grid_shape: tuple[int, int]
    sigma_floor: float = DEFAULT_SIGMA_FLOOR
    iv_floor: float = DEFAULT_IV_FLOOR

    mean: np.ndarray = field(init=False)
    std: np.ndarray = field(init=False)
    count: np.ndarray = field(init=False)
    _m2: np.ndarray = field(init=False)
    fitted: bool = field(init=False, default=False)

    def __post_init__(self):
        """Initialize running statistics arrays for the configured grid shape."""
        grid_i, grid_j = self.grid_shape
        self.mean = np.zeros((grid_i, grid_j))
        self._m2 = np.zeros((grid_i, grid_j))
        self.count = np.zeros((grid_i, grid_j), int)
        self.std = np.full((grid_i, grid_j), np.nan)

    def _check(self, x):
        x = _as_3d(x)
        if x.shape[1:] != self.grid_shape:
            raise ValueError(f"Shape {x.shape[1:]} != {self.grid_shape}")
        return x

    def fit(self, iv):
        """Fit statistics from a full batch of IV surfaces."""
        x = iv_to_log_iv(self._check(iv), self.iv_floor)
        m = np.isfinite(x)
        c = m.sum(0)

        masked = np.where(m, x, 0)
        mean = np.divide(masked.sum(0), np.maximum(c, 1))

        dev = np.where(m, x - mean, 0)
        m2 = (dev * dev).sum(0)

        self.mean, self._m2, self.count = mean, m2, c
        self._finalize()
        return self

    def partial_fit(self, iv):
        """Update running statistics with one or more additional surfaces."""
        # Welford's update https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
        x = iv_to_log_iv(self._check(iv), self.iv_floor)
        m = np.isfinite(x)

        for i in range(x.shape[0]):
            mask = m[i]
            if not mask.any():
                continue
            xi = x[i]
            new_c = self.count + mask

            delta = np.where(mask, xi - self.mean, 0)
            self.mean += np.where(mask, delta / np.maximum(new_c, 1), 0)
            delta2 = np.where(mask, xi - self.mean, 0)
            self._m2 += np.where(mask, delta * delta2, 0)

            self.count = new_c

        self._finalize()
        return self

    def _finalize(self):
        var = np.where(self.count > 1, self._m2 / (self.count - 1), np.nan)
        std = np.sqrt(var)
        self.std = np.where((std > self.sigma_floor) & np.isfinite(std), std, self.sigma_floor)
        self.fitted = self.count.any()

    def _check_fitted(self):
        if not self.fitted:
            raise RuntimeError("Not fitted")

    def transform(self, iv):
        """Normalize IV surfaces into z-space using fitted statistics."""
        self._check_fitted()
        x = np.asarray(iv, float)
        squeeze = x.ndim == 2
        x = x[None] if squeeze else x

        if x.shape[-2:] != self.grid_shape:
            raise ValueError("Shape mismatch")

        z = (iv_to_log_iv(x, self.iv_floor) - self.mean) / self.std
        return z[0] if squeeze else z

    def inverse_transform(self, z, return_log_iv=False):
        """Map normalized z-space surfaces back to log-IV or IV space."""
        self._check_fitted()
        z = np.asarray(z, float)

        if z.shape[-2:] != self.grid_shape:
            raise ValueError("Shape mismatch")

        log_iv = z * self.std + self.mean
        return log_iv if return_log_iv else np.exp(log_iv)

    normalize = transform
    denormalize = inverse_transform

    def save(self, path):
        """Persist fitted normalization statistics to a ``.npz`` file."""
        self._check_fitted()
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            p,
            mean=self.mean,
            std=self.std,
            count=self.count,
            m2=self._m2,
            grid_shape=self.grid_shape,
            sigma_floor=self.sigma_floor,
            iv_floor=self.iv_floor,
        )

    @classmethod
    def load(cls, path):
        """Load a persisted normalizer from disk."""
        d = np.load(Path(path))
        obj = cls(tuple(d["grid_shape"]), float(d["sigma_floor"]), float(d["iv_floor"]))
        obj.mean = d["mean"]
        obj.std = d["std"]
        obj.count = d["count"]
        obj._m2 = d["m2"]
        obj.fitted = obj.count.any()
        return obj


def normalize_surface(iv, mean, std, iv_floor=DEFAULT_IV_FLOOR):
    """Normalize a surface using explicit mean/std arrays."""
    iv = np.asarray(iv, float)
    if iv.shape[-2:] != mean.shape:
        raise ValueError("Shape mismatch")
    return (iv_to_log_iv(iv, iv_floor) - mean) / std


def denormalize_surface(z, mean, std, return_log_iv=False):
    """Invert normalization using explicit mean/std arrays."""
    z = np.asarray(z, float)
    if z.shape[-2:] != mean.shape:
        raise ValueError("Shape mismatch")
    log_iv = z * std + mean
    return log_iv if return_log_iv else np.exp(log_iv)
