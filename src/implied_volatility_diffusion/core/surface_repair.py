"""Post-process implied-volatility surfaces for smoothness and no-arbitrage."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter

from implied_volatility_diffusion.arbitrage import check_iv_surface_arbitrage
from implied_volatility_diffusion.arbitrage_checks.checks import _bs_call_grid
from implied_volatility_diffusion.pricing.implied_vol import implied_volatility


def repair_calendar_monotone(iv: np.ndarray, tau: np.ndarray) -> np.ndarray:
    """Enforce calendar monotonicity in total variance along ``tau``."""
    iv_arr = np.asarray(iv, dtype=float)
    t = np.asarray(tau, dtype=float).reshape(-1)
    if iv_arr.shape[-1] != t.size:
        raise ValueError(f"iv last axis ({iv_arr.shape[-1]}) must match len(tau)={t.size}")
    if t.size < 2:
        return iv_arr.copy()

    w = (iv_arr**2) * t
    valid = np.isfinite(w) & (iv_arr > 0.0)
    w_fill = np.where(valid, w, -np.inf)
    w_mono = np.maximum.accumulate(w_fill, axis=-1)
    w_mono = np.where(np.isfinite(w_mono), w_mono, w)

    with np.errstate(invalid="ignore", divide="ignore"):
        iv_new = np.sqrt(np.maximum(w_mono, 0.0) / t)
    return np.where(valid, iv_new, iv_arr)


def _isotonic_non_decreasing(y: np.ndarray) -> np.ndarray:
    """L2 isotonic regression (non-decreasing), unweighted PAV."""
    vals = np.asarray(y, dtype=np.float64).ravel()
    n = vals.size
    if n == 0:
        return vals.copy()
    if n == 1:
        return vals.copy()

    block_vals: list[float] = []
    block_sizes: list[int] = []
    for v in vals:
        block_vals.append(float(v))
        block_sizes.append(1)
        while len(block_vals) >= 2 and block_vals[-2] > block_vals[-1]:
            n2 = block_sizes.pop()
            v2 = block_vals.pop()
            n1 = block_sizes.pop()
            v1 = block_vals.pop()
            merged_n = n1 + n2
            block_vals.append((v1 * n1 + v2 * n2) / merged_n)
            block_sizes.append(merged_n)

    out = np.empty(n, dtype=np.float64)
    idx = 0
    for v, cnt in zip(block_vals, block_sizes, strict=True):
        out[idx : idx + cnt] = v
        idx += cnt
    return out


def repair_wing_monotonicity(
    iv_slice: np.ndarray,
    *,
    sigma_lo: float = 1e-4,
    sigma_hi: float = 10.0,
) -> np.ndarray:
    """Enforce non-decreasing IV away from the smile minimum per tau column."""
    out = np.asarray(iv_slice, dtype=float).copy()
    n_m, n_t = out.shape

    lo_fence = sigma_lo * 2.0
    hi_fence = sigma_hi * 0.9

    for j in range(n_t):
        col = out[:, j]
        reliable = np.isfinite(col) & (col > lo_fence) & (col < hi_fence)

        if reliable.sum() < 2:
            finite = np.isfinite(col) & (col > 0)
            if finite.sum() < 2:
                continue
            reliable = finite

        vals = np.where(reliable, col, np.inf)
        min_idx = int(np.argmin(vals))

        for i in range(min_idx - 1, -1, -1):
            if not reliable[i] or col[i] < col[i + 1]:
                col[i] = col[i + 1]
        for i in range(min_idx + 1, n_m):
            if not reliable[i] or col[i] < col[i - 1]:
                col[i] = col[i - 1]

    return out


def _smooth_log_iv(
    iv: np.ndarray,
    *,
    sigma_log_moneyness: float,
    sigma_tau: float,
    iv_floor: float,
) -> np.ndarray:
    """Separable Gaussian smooth on ``log(iv)``; finite cells only."""
    if sigma_log_moneyness <= 0.0 and sigma_tau <= 0.0:
        return np.asarray(iv, dtype=float).copy()

    arr = np.asarray(iv, dtype=float)
    finite = np.isfinite(arr) & (arr > iv_floor)
    if not np.any(finite):
        return arr.copy()

    log_iv = np.where(finite, np.log(np.maximum(arr, iv_floor)), 0.0)
    weight = finite.astype(np.float64)
    sm_log = gaussian_filter(log_iv * weight, sigma=(sigma_log_moneyness, sigma_tau), mode="nearest")
    sm_w = gaussian_filter(weight, sigma=(sigma_log_moneyness, sigma_tau), mode="nearest")
    with np.errstate(invalid="ignore", divide="ignore"):
        smoothed_log = np.where(sm_w > 1e-12, sm_log / sm_w, np.nan)
    out = np.where(finite, np.exp(smoothed_log), arr)
    return np.where(np.isfinite(out), out, arr)


def _call_bounds(
    moneyness: np.ndarray,
    tau: float,
    *,
    spot: float,
    rate: float,
    dividend_yield: float,
) -> tuple[np.ndarray, np.ndarray]:
    k = np.asarray(moneyness, dtype=float) * float(spot)
    t = float(tau)
    upper = float(spot) * np.exp(-float(dividend_yield) * t)
    lower = np.maximum(
        float(spot) * np.exp(-float(dividend_yield) * t) - k * np.exp(-float(rate) * t),
        0.0,
    )
    return lower, np.full(k.shape, upper, dtype=float)


def _repair_convex_calls(
    prices: np.ndarray,
    strikes: np.ndarray,
    *,
    lower: np.ndarray,
    upper: np.ndarray,
) -> np.ndarray:
    """Project call prices onto a convex, bound-respecting slice along strike."""
    c = np.asarray(prices, dtype=float).copy()
    k = np.asarray(strikes, dtype=float).reshape(-1)
    if c.size != k.size or c.size < 2:
        return np.clip(c, lower, upper)

    valid = np.isfinite(c) & np.isfinite(k) & (k > 0.0)
    if valid.sum() < 2:
        return np.where(valid, np.clip(c, lower, upper), c)

    c = np.clip(c, lower, upper)
    dk = np.diff(k)
    if np.any(dk <= 0.0):
        raise ValueError("strikes must be strictly increasing")

    slopes = np.diff(c) / dk
    # Calls decrease in strike: slopes <= 0; convexity: slopes non-decreasing in K.
    slopes_iso = _isotonic_non_decreasing(np.minimum(slopes, 0.0))

    rebuilt = np.empty_like(c)
    rebuilt[0] = c[0]
    for i in range(slopes_iso.size):
        rebuilt[i + 1] = rebuilt[i] + slopes_iso[i] * dk[i]
    return np.clip(rebuilt, lower, upper)


def _iv_from_calls(
    calls: np.ndarray,
    moneyness: np.ndarray,
    tau: float,
    *,
    spot: float,
    rate: float,
    dividend_yield: float,
    iv_floor: float,
    sigma_hi: float,
) -> np.ndarray:
    """Invert clipped call prices back to implied vol on one maturity column."""
    m = np.asarray(moneyness, dtype=float).reshape(-1)
    c = np.asarray(calls, dtype=float).reshape(-1)
    out = np.full(m.size, np.nan, dtype=float)
    t = float(tau)
    if t <= 0.0:
        return out

    for i, mi in enumerate(m):
        if not np.isfinite(c[i]):
            continue
        strike = float(mi * spot)
        try:
            out[i] = implied_volatility(
                float(c[i]),
                float(spot),
                strike,
                t,
                float(rate),
                dividend_yield=float(dividend_yield),
                sigma_lo=iv_floor,
                sigma_hi=sigma_hi,
            )
        except (ValueError, ArithmeticError):
            out[i] = np.nan
    return out


def _blend_arrays(
    original: np.ndarray,
    adjusted: np.ndarray,
    strength: float,
    *,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """Convex mix ``(1-strength)*original + strength*adjusted`` on ``mask`` (default: finite)."""
    w = float(np.clip(strength, 0.0, 1.0))
    if w <= 0.0:
        return np.asarray(original, dtype=float).copy()
    if w >= 1.0:
        return np.asarray(adjusted, dtype=float).copy()
    orig = np.asarray(original, dtype=float)
    adj = np.asarray(adjusted, dtype=float)
    use = np.isfinite(orig) & np.isfinite(adj) if mask is None else mask
    out = orig.copy()
    out[use] = (1.0 - w) * orig[use] + w * adj[use]
    return out


def repair_butterfly_convex(
    iv: np.ndarray,
    moneyness: np.ndarray,
    tau: np.ndarray,
    *,
    spot: float,
    rate: float,
    dividend_yield: float = 0.0,
    iv_floor: float = 1e-4,
    sigma_hi: float = 10.0,
    repair_wings: bool = True,
    strength: float = 1.0,
) -> np.ndarray:
    """Repair butterfly violations via convex call-price projection per maturity."""
    iv_arr = np.asarray(iv, dtype=float).copy()
    m = np.asarray(moneyness, dtype=float).reshape(-1)
    t = np.asarray(tau, dtype=float).reshape(-1)

    if repair_wings:
        iv_arr = repair_wing_monotonicity(iv_arr, sigma_lo=iv_floor, sigma_hi=sigma_hi)

    strikes = m * float(spot)
    for j, tj in enumerate(t):
        if tj <= 0.0:
            continue
        col = iv_arr[:, j]
        if not np.any(np.isfinite(col) & (col > iv_floor)):
            continue
        calls_col = _bs_call_grid(col[:, None], m, np.array([tj]), spot=spot, rate=rate, dividend_yield=dividend_yield)[
            :, 0
        ]
        lower, upper = _call_bounds(m, tj, spot=spot, rate=rate, dividend_yield=dividend_yield)
        projected = _repair_convex_calls(calls_col, strikes, lower=lower, upper=upper)
        blended_calls = _blend_arrays(calls_col, projected, strength)
        iv_arr[:, j] = _iv_from_calls(
            blended_calls,
            m,
            tj,
            spot=spot,
            rate=rate,
            dividend_yield=dividend_yield,
            iv_floor=iv_floor,
            sigma_hi=sigma_hi,
        )
    return iv_arr


def volgan_generative_repair_settings(
    *,
    tol: float = 1e-6,
    iv_floor: float = 1e-4,
    sigma_hi: float = 10.0,
    max_iterations: int = 6,
    smooth_sigma_log_moneyness: float = 0.6,
    smooth_sigma_tau: float = 0.4,
    blend: float = 1.0,
    butterfly_strength: float = 1.0,
) -> "SurfaceRepairSettings":
    """Repair settings for generative IV surfaces (VolGAN-style).

    Runs smooth / calendar / butterfly projection only when arbitrage checks fail,
    leaving clean diffusion samples unchanged.
    """
    return SurfaceRepairSettings(
        iv_floor=iv_floor,
        sigma_hi=sigma_hi,
        tol=tol,
        max_iterations=max_iterations,
        smooth_sigma_log_moneyness=smooth_sigma_log_moneyness,
        smooth_sigma_tau=smooth_sigma_tau,
        repair_calendar=True,
        repair_butterfly=True,
        repair_wings=True,
        only_if_violated=True,
        blend=blend,
        butterfly_strength=butterfly_strength,
    )


@dataclass(frozen=True)
class SurfaceRepairSettings:
    """Options for :func:`repair_iv_surface`."""

    iv_floor: float = 1e-4
    sigma_hi: float = 10.0
    tol: float = 1e-8
    max_iterations: int = 6
    smooth_sigma_log_moneyness: float = 0.6
    smooth_sigma_tau: float = 0.4
    repair_calendar: bool = True
    repair_butterfly: bool = True
    repair_wings: bool = True
    only_if_violated: bool = False
    blend: float = 1.0
    butterfly_strength: float = 1.0


def repair_iv_surface(
    iv: np.ndarray,
    moneyness: np.ndarray,
    tau: np.ndarray,
    *,
    spot: float,
    rate: float,
    dividend_yield: float = 0.0,
    settings: SurfaceRepairSettings | None = None,
) -> np.ndarray:
    """Smooth and project an IV surface toward no-arbitrage.

    Pipeline (iterated until checks pass or ``max_iterations``):

    1. Clip IV to ``iv_floor``.
    2. Optional Gaussian smooth on ``log(iv)``.
    3. Calendar repair via monotone total variance.
    4. Butterfly repair via convex call-price projection per maturity.
    """
    cfg = settings or SurfaceRepairSettings()
    original = np.asarray(iv, dtype=float).copy()
    out = original.copy()
    m = np.asarray(moneyness, dtype=float).reshape(-1)
    t = np.asarray(tau, dtype=float).reshape(-1)
    report = check_iv_surface_arbitrage(
        out,
        m,
        t,
        spot=float(spot),
        rate=float(rate),
        dividend_yield=float(dividend_yield),
        tol=float(cfg.tol),
    )

    for _ in range(max(1, int(cfg.max_iterations))):
        finite = np.isfinite(out) & (out > 0.0)
        out = np.where(finite, np.maximum(out, cfg.iv_floor), out)

        if cfg.smooth_sigma_log_moneyness > 0.0 or cfg.smooth_sigma_tau > 0.0:
            if not cfg.only_if_violated or not report.arbitrage_free:
                smoothed = _smooth_log_iv(
                    out,
                    sigma_log_moneyness=cfg.smooth_sigma_log_moneyness,
                    sigma_tau=cfg.smooth_sigma_tau,
                    iv_floor=cfg.iv_floor,
                )
                out = _blend_arrays(out, smoothed, cfg.blend, mask=finite)

        if cfg.repair_calendar and (not cfg.only_if_violated or not report.calendar_ok):
            cal = repair_calendar_monotone(out, t)
            out = _blend_arrays(out, cal, cfg.blend, mask=finite)

        report = check_iv_surface_arbitrage(
            out,
            m,
            t,
            spot=float(spot),
            rate=float(rate),
            dividend_yield=float(dividend_yield),
            tol=float(cfg.tol),
        )

        if cfg.repair_butterfly and (not cfg.only_if_violated or not report.butterfly_ok):
            bfly = repair_butterfly_convex(
                out,
                m,
                t,
                spot=spot,
                rate=rate,
                dividend_yield=dividend_yield,
                iv_floor=cfg.iv_floor,
                sigma_hi=cfg.sigma_hi,
                repair_wings=cfg.repair_wings,
                strength=cfg.butterfly_strength,
            )
            out = _blend_arrays(out, bfly, cfg.blend, mask=finite)

        report = check_iv_surface_arbitrage(
            out,
            m,
            t,
            spot=float(spot),
            rate=float(rate),
            dividend_yield=float(dividend_yield),
            tol=float(cfg.tol),
        )
        if report.arbitrage_free:
            break

    if cfg.repair_calendar and (not cfg.only_if_violated or not report.calendar_ok):
        cal = repair_calendar_monotone(out, t)
        out = _blend_arrays(out, cal, cfg.blend, mask=np.isfinite(out))

    if cfg.blend >= 1.0:
        return out
    original = np.asarray(iv, dtype=float)
    return _blend_arrays(original, out, cfg.blend, mask=np.isfinite(original) & np.isfinite(out))


def repair_iv_surfaces(
    iv: np.ndarray,
    moneyness: np.ndarray,
    tau: np.ndarray,
    *,
    spot: float,
    rate: float,
    dividend_yield: float = 0.0,
    settings: SurfaceRepairSettings | None = None,
) -> np.ndarray:
    """Apply :func:`repair_iv_surface` to every ``(..., M, T)`` slice."""
    iv_t = np.asarray(iv, dtype=float)
    leading = iv_t.shape[:-2]
    flat = iv_t.reshape(-1, iv_t.shape[-2], iv_t.shape[-1])
    repaired = np.stack(
        [
            repair_iv_surface(
                flat[i],
                moneyness,
                tau,
                spot=spot,
                rate=rate,
                dividend_yield=dividend_yield,
                settings=settings,
            )
            for i in range(flat.shape[0])
        ],
        axis=0,
    )
    return repaired.reshape(leading + repaired.shape[-2:])


__all__ = [
    "SurfaceRepairSettings",
    "repair_butterfly_convex",
    "repair_calendar_monotone",
    "repair_iv_surface",
    "repair_iv_surfaces",
    "repair_wing_monotonicity",
    "volgan_generative_repair_settings",
]