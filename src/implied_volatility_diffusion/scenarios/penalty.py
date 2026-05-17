"""Scalar arbitrage penalty Φ(σ) aligned with :class:`~diffusion.arbitrage_torch.ArbitragePenalty`."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from implied_volatility_diffusion.arbitrage_checks.checks import _bs_call_grid


@dataclass(frozen=True)
class SurfaceArbitrageWeights:
    """Per-family multipliers; set to 0 to disable a family."""

    calendar: float = 1.0
    butterfly: float = 1.0
    call: float = 1.0


@dataclass(frozen=True)
class SurfaceArbitragePenalty:
    """Evaluate Φ(σ) on IV grids using the same relu-mean penalties as training."""

    moneyness: np.ndarray
    tau: np.ndarray
    spot: float = 1.0
    rate: float = 0.0
    dividend_yield: float = 0.0
    weights: SurfaceArbitrageWeights | None = None

    def __post_init__(self) -> None:
        m = np.asarray(self.moneyness, dtype=float).reshape(-1)
        t = np.asarray(self.tau, dtype=float).reshape(-1)
        if m.size < 1 or t.size < 1:
            raise ValueError("moneyness and tau must be non-empty")
        if bool(np.any(np.diff(m) <= 0.0)):
            raise ValueError("moneyness must be strictly increasing")
        if bool(np.any(np.diff(t) <= 0.0)):
            raise ValueError("tau must be strictly increasing")
        object.__setattr__(self, "moneyness", m)
        object.__setattr__(self, "tau", t)
        object.__setattr__(self, "strikes", m * float(self.spot))

    @property
    def grid_shape(self) -> tuple[int, int]:
        return int(self.moneyness.size), int(self.tau.size)

    def _ensure_iv(self, iv: np.ndarray) -> np.ndarray:
        iv_arr = np.asarray(iv, dtype=float)
        if iv_arr.shape[-2:] != self.grid_shape:
            raise ValueError(f"IV trailing shape {iv_arr.shape[-2:]} must match grid {self.grid_shape}")
        return iv_arr

    def call_prices(self, iv: np.ndarray) -> np.ndarray:
        iv_arr = self._ensure_iv(iv)
        return _bs_call_grid(
            iv_arr,
            self.moneyness,
            self.tau,
            spot=self.spot,
            rate=self.rate,
            dividend_yield=self.dividend_yield,
        )

    def calendar_penalty(self, iv: np.ndarray) -> float:
        iv_arr = self._ensure_iv(iv)
        if iv_arr.shape[-1] < 2:
            return 0.0
        w = (iv_arr * iv_arr) * self.tau.reshape(1, -1)
        diff = np.diff(w, axis=-1)
        return float(np.mean(np.maximum(0.0, -diff)))

    def butterfly_penalty(self, call_prices: np.ndarray) -> float:
        c = np.asarray(call_prices, dtype=float)
        k = self.strikes
        if k.size < 3:
            return 0.0
        dk_l = (k[1:-1] - k[:-2]).reshape(-1, 1)
        dk_r = (k[2:] - k[1:-1]).reshape(-1, 1)
        denom = dk_l * dk_r * (dk_l + dk_r) / 2.0
        d2 = (dk_l * c[2:, :] - (dk_l + dk_r) * c[1:-1, :] + dk_r * c[:-2, :]) / denom
        return float(np.mean(np.maximum(0.0, -d2)))

    def call_penalty(self, call_prices: np.ndarray) -> float:
        c = np.asarray(call_prices, dtype=float)
        k = self.strikes
        if k.size < 2:
            return 0.0
        dk = (k[1:] - k[:-1]).reshape(-1, 1)
        slope = np.diff(c, axis=0) / dk
        return float(np.mean(np.maximum(0.0, slope)))

    def forward(self, iv: np.ndarray) -> dict[str, float]:
        w = self.weights or SurfaceArbitrageWeights()
        iv_arr = self._ensure_iv(iv)
        need_calls = w.butterfly > 0.0 or w.call > 0.0
        c = self.call_prices(iv_arr) if need_calls else np.zeros(iv_arr.shape)
        return {
            "calendar": self.calendar_penalty(iv_arr) * w.calendar,
            "butterfly": self.butterfly_penalty(c) * w.butterfly,
            "call": self.call_penalty(c) * w.call,
        }

    def __call__(self, iv: np.ndarray) -> float:
        """Total penalty Φ(σ); zero when the surface is arbitrage-free."""
        parts = self.forward(iv)
        return float(sum(parts.values()))

    def batch(self, iv: np.ndarray) -> np.ndarray:
        """Per-scenario penalties for ``(..., M, T)`` IV batches."""
        iv_arr = np.asarray(iv, dtype=float)
        leading = iv_arr.shape[:-2]
        flat = iv_arr.reshape(-1, *self.grid_shape)
        out = np.array([self(s) for s in flat], dtype=float)
        return out.reshape(leading) if leading else out.reshape(())
