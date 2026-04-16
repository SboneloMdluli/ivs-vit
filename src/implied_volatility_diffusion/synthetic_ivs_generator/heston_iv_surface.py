"""Heston COS + LHS configuration helpers (wraps generic :mod:`iv_surface` utilities)."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from ivs_config import merge_config_files

from implied_volatility_diffusion.iv_surface import (
    grid_axes,
    implied_vol_surface_on_grid,
    implied_vol_surfaces_from_param_matrix,
    lhs_params_from_config,
    lhs_params_multi_batch_from_config,
)
from implied_volatility_diffusion.synthetic_ivs_generator.black_scholes import implied_volatility
from implied_volatility_diffusion.synthetic_ivs_generator.heston_cos import heston_call_cos
from implied_volatility_diffusion.synthetic_ivs_generator.heston_iv_goals import (
    HESTON_GOAL_YAML,
    HestonIvGoal,
    coerce_heston_iv_goal,
)

HESTON_PARAM_ORDER = ("v0", "rho", "sigma", "theta", "kappa", "r")

HESTON_IV_SURFACE_YAML = "heston_iv_surface.yaml"
IV_SURFACE_GRID_YAML = "iv_surface_grid.yaml"


def load_heston_iv_surface_config(config_dir: str | Path) -> dict[str, Any]:
    d = Path(config_dir)
    return merge_config_files(d / HESTON_IV_SURFACE_YAML, d / IV_SURFACE_GRID_YAML)


def load_heston_iv_surface_goal_config(
    config_dir: str | Path,
    goal: HestonIvGoal | str,
) -> dict[str, Any]:
    """Load base Heston + grid YAML, then merge a **goal** overlay (low vol, skew, etc.).

    ``goal`` is a :class:`HestonIvGoal` member, or the same string as its ``value`` (e.g.
    ``\"low_vol\"``). Later keys in the merge win on conflicts; the overlay only sets
    ranges and related keys for that scenario.
    """
    g = coerce_heston_iv_goal(goal)
    d = Path(config_dir)
    return merge_config_files(
        d / HESTON_IV_SURFACE_YAML,
        d / IV_SURFACE_GRID_YAML,
        d / HESTON_GOAL_YAML[g],
    )


def _heston_cos_section(cfg: Mapping[str, Any]) -> Mapping[str, Any]:
    if "heston_cos_pricer" in cfg:
        return cfg["heston_cos_pricer"]
    return cfg.get("cos", {})


def lhs_heston_params(
    cfg: Mapping[str, Any],
    *,
    n_samples: int | None = None,
    seed: int | None = None,
) -> np.ndarray:
    """Latin Hypercube sample of Heston parameters; shape ``(n, 6)`` in ``HESTON_PARAM_ORDER``."""
    return lhs_params_from_config(
        cfg,
        param_order=HESTON_PARAM_ORDER,
        ranges_key="heston_ranges",
        n_samples=n_samples,
        seed=seed,
    )


def lhs_heston_params_multi_batch(
    cfg: Mapping[str, Any],
    *,
    n_samples: int | None = None,
    n_batches: int | None = None,
    seed: int | None = None,
    seed_stride: int | None = None,
) -> np.ndarray:
    """Several independent LHS batches over ``cfg['heston_ranges']``."""
    return lhs_params_multi_batch_from_config(
        cfg,
        param_order=HESTON_PARAM_ORDER,
        ranges_key="heston_ranges",
        n_samples=n_samples,
        n_batches=n_batches,
        seed=seed,
        seed_stride=seed_stride,
    )


def implied_vol_surface_for_params(
    params: np.ndarray,
    cfg: Mapping[str, Any],
    *,
    spot: float | None = None,
    inst_var: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Implied-vol grid for one Heston parameter vector (COS prices + BS inversion).

    By default the spot and initial variance come from ``cfg['market']['spot']`` and
    ``params[0]`` (``v0`` in :data:`HESTON_PARAM_ORDER`). For sequential surfaces along
    a simulated path, pass ``spot`` and ``inst_var`` for the state at the observation time.
    """
    market = cfg["market"]
    spot_eff = float(market["spot"]) if spot is None else float(spot)
    q = float(market.get("dividend_yield", 0.0))
    m, tau = grid_axes(cfg)

    v0, rho, sigma_h, theta, kappa, r = (float(x) for x in params)
    v_price = float(v0) if inst_var is None else float(inst_var)
    cos_cfg = _heston_cos_section(cfg)
    n_terms_base = int(cos_cfg.get("n_terms", 1024))
    L = float(cos_cfg.get("truncation_L", 14.0))
    tau_ref = float(cos_cfg.get("short_tau_tau_ref", 0.25))
    n_terms_max = int(cos_cfg.get("n_terms_max", 4096))

    iv_cfg = cfg.get("implied_vol", {})
    iv_opts = {
        "sigma_lo": float(iv_cfg.get("sigma_lo", 1e-4)),
        "sigma_hi": float(iv_cfg.get("sigma_hi", 1.0)),
        "xtol": float(iv_cfg.get("brent_xtol", 1e-8)),
        "newton_refinement_steps": int(
            iv_cfg.get("newton_refinement_steps", iv_cfg.get("newton_max_iter", 3))
        ),
        "newton_tol": float(iv_cfg.get("newton_tol", 1e-10)),
        "jackel_iterations": int(iv_cfg.get("jackel_iterations", 0)),
        "vega_floor_scale": float(iv_cfg.get("vega_floor_scale", 1e-14)),
    }

    def model_call_price(strike: float, ttm: float) -> float:
        scale = max(1.0, math.sqrt(tau_ref / max(ttm, 1e-12)))
        n_terms = int(min(n_terms_max, round(n_terms_base * scale)))
        return float(
            heston_call_cos(
                spot_eff,
                strike,
                ttm,
                r,
                kappa,
                theta,
                sigma_h,
                rho,
                v_price,
                dividend_yield=q,
                n_terms=n_terms,
                truncation_L=L,
            )
        )

    iv = implied_vol_surface_on_grid(
        m,
        tau,
        spot=spot_eff,
        rate=r,
        dividend_yield=q,
        model_call_price=model_call_price,
        to_implied_vol=implied_volatility,
        implied_vol_options=iv_opts,
    )

    tau_arr = np.asarray(tau, dtype=float)
    tau_thr = float(iv_cfg.get("tau_extrapolate_below", float("nan")))
    if math.isfinite(tau_thr) and tau_thr > 0.0 and iv.shape[1] > 0:
        j_ref = int(np.searchsorted(tau_arr, tau_thr, side="left"))
        if 0 <= j_ref < iv.shape[1]:
            for j, tj in enumerate(tau_arr):
                if float(tj) + 1e-12 < tau_thr:
                    iv[:, j] = iv[:, j_ref]

    return m, tau, iv


def _heston_spot_variance_euler_step(
    s: float,
    v: float,
    dt: float,
    r: float,
    q: float,
    kappa: float,
    theta: float,
    sigma_v: float,
    rho: float,
    rng: np.random.Generator,
) -> tuple[float, float]:
    """Full-truncation Euler step for risk-neutral Heston (log-price formulation on spot level)."""
    z1, z2i = rng.standard_normal(2)
    rho_c = max(-1.0, min(1.0, float(rho)))
    z2 = rho_c * z1 + math.sqrt(max(1e-18, 1.0 - rho_c * rho_c)) * z2i
    sqrt_dt = math.sqrt(float(dt))
    v_pos = max(float(v), 0.0)
    sqrt_v = math.sqrt(v_pos)
    v_next = v_pos + kappa * (theta - v_pos) * float(dt) + sigma_v * sqrt_v * sqrt_dt * z2
    v_next = max(v_next, 0.0)
    s_next = float(s) + (r - q) * float(s) * float(dt) + sqrt_v * float(s) * sqrt_dt * z1
    s_next = max(s_next, 1e-12)
    return s_next, v_next


def implied_vol_surfaces_lhs(
    cfg: Mapping[str, Any],
    *,
    n_samples: int | None = None,
    n_batches: int | None = None,
    seed: int | None = None,
    seed_stride: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Latin-hypercube Heston draws and one implied-vol surface per draw."""
    lhs_cfg = cfg.get("lhs", {})
    nb = int(n_batches if n_batches is not None else lhs_cfg.get("n_batches", 1))
    if nb <= 1:
        params = lhs_heston_params(cfg, n_samples=n_samples, seed=seed)
    else:
        params = lhs_heston_params_multi_batch(
            cfg,
            n_samples=n_samples,
            n_batches=nb,
            seed=seed,
            seed_stride=seed_stride,
        )

    def build_surface(row: np.ndarray, c: Mapping[str, Any]) -> np.ndarray:
        _, _, surf = implied_vol_surface_for_params(row, c)
        return surf

    return implied_vol_surfaces_from_param_matrix(params, cfg, build_surface=build_surface)


def implied_vol_surfaces_sequential_lhs(
    cfg: Mapping[str, Any],
    *,
    n_samples: int | None = None,
    n_batches: int | None = None,
    seed: int | None = None,
    seed_stride: int | None = None,
    n_steps: int | None = None,
    dt: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Latin-hypercube Heston draws and a short **sequence** of IV surfaces per draw.

    For each parameter row, spot and variance follow a risk-neutral Heston Euler path
    (correlated Brownian increments). At each time index the surface is built with
    :func:`implied_vol_surface_for_params` using the current ``(spot, variance)`` and
    the same ``(kappa, theta, sigma, rho, r)`` as in the COS pricer.

    Shape of the implied-vol array is ``(n_paths, n_steps, n_moneyness, n_tau)``.

    Config (optional) under ``sequential_ivs``:

    - ``n_steps``: number of surfaces per path (default ``8``).
    - ``dt``: year fraction between snapshots (default ``1/252``).
    - ``path_seed_stride``: offset between path RNG seeds (default ``100_000``).
    """
    seq_cfg = cfg.get("sequential_ivs") or {}
    n_st = int(n_steps if n_steps is not None else seq_cfg.get("n_steps", 8))
    if n_st < 1:
        raise ValueError("n_steps must be >= 1")
    dt_eff = float(dt if dt is not None else seq_cfg.get("dt", 1.0 / 252.0))
    if dt_eff <= 0.0:
        raise ValueError("dt must be positive")
    path_stride = int(seq_cfg.get("path_seed_stride", 100_000))

    lhs_cfg = cfg.get("lhs", {})
    nb = int(n_batches if n_batches is not None else lhs_cfg.get("n_batches", 1))
    if nb <= 1:
        params = lhs_heston_params(cfg, n_samples=n_samples, seed=seed)
    else:
        params = lhs_heston_params_multi_batch(
            cfg,
            n_samples=n_samples,
            n_batches=nb,
            seed=seed,
            seed_stride=seed_stride,
        )

    m, tau = grid_axes(cfg)
    market = cfg["market"]
    s0 = float(market["spot"])
    q = float(market.get("dividend_yield", 0.0))

    lhs_seed = int(seed if seed is not None else lhs_cfg["seed"])
    n_paths = params.shape[0]
    iv = np.empty((n_paths, n_st, m.size, tau.size), dtype=float)

    for p in range(n_paths):
        row = params[p]
        v0, rho, sigma_h, theta, kappa, r = (float(x) for x in row)
        rng = np.random.default_rng(lhs_seed + (p + 1) * path_stride)
        s_cur, v_cur = s0, v0
        for k in range(n_st):
            _, _, surf = implied_vol_surface_for_params(row, cfg, spot=s_cur, inst_var=v_cur)
            iv[p, k, :, :] = surf
            if k < n_st - 1:
                s_cur, v_cur = _heston_spot_variance_euler_step(
                    s_cur,
                    v_cur,
                    dt_eff,
                    r,
                    q,
                    kappa,
                    theta,
                    sigma_h,
                    rho,
                    rng,
                )

    return params, m, tau, iv
