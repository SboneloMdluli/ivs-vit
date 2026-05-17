"""VolGAN-style exponential scenario weights from arbitrage penalties."""

from __future__ import annotations

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]


def volgan_exponential_weights(penalties: np.ndarray, beta: float) -> np.ndarray:
    """Return w_i = exp(-β Φ_i) / Σ_j exp(-β Φ_j) (VolGAN reweighting).

    Args:
        penalties: Per-scenario arbitrage penalties Φ(σ^i), shape ``(N,)`` or broadcastable.
        beta: Sensitivity; larger β concentrates mass on low-penalty scenarios.

    Returns:
        Normalized weights summing to 1.
    """
    if beta < 0.0:
        raise ValueError("beta must be non-negative")
    phi = np.asarray(penalties, dtype=float).reshape(-1)
    if phi.size == 0:
        raise ValueError("penalties must be non-empty")
    if beta == 0.0:
        return np.full(phi.shape, 1.0 / phi.size, dtype=float)
    log_w = -beta * phi
    log_w -= np.max(log_w)
    w = np.exp(log_w)
    return w / w.sum()


def volgan_exponential_weights_torch(penalties: "torch.Tensor", beta: float) -> "torch.Tensor":
    """Torch variant of :func:`volgan_exponential_weights` using ``softmax(-β Φ)``."""
    if torch is None:
        raise ImportError("torch is required for volgan_exponential_weights_torch")
    if beta < 0.0:
        raise ValueError("beta must be non-negative")
    phi = penalties.reshape(-1)
    if phi.numel() == 0:
        raise ValueError("penalties must be non-empty")
    if beta == 0.0:
        return torch.full_like(phi, 1.0 / phi.numel())
    return torch.softmax(-beta * phi, dim=0)
