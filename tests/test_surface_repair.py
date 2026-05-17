"""Tests for IV surface smoothness and no-arbitrage repair."""

import numpy as np

from implied_volatility_diffusion import (
    SurfaceRepairSettings,
    check_iv_surface_arbitrage,
    repair_calendar_monotone,
    repair_iv_surface,
    volgan_generative_repair_settings,
)
from implied_volatility_diffusion.core.surface_repair import repair_butterfly_convex


def _flat_bs_surface(moneyness: np.ndarray, tau: np.ndarray, sigma: float) -> np.ndarray:
    return np.full((moneyness.size, tau.size), float(sigma), dtype=float)


def test_calendar_repair_restores_monotone_total_variance() -> None:
    m = np.linspace(0.5, 2.5, 21)
    tau = np.array([0.25, 0.5, 1.0])
    iv = np.empty((m.size, tau.size), dtype=float)
    iv[:, 0] = 0.40
    iv[:, 1] = 0.20
    iv[:, 2] = 0.20

    rep = repair_calendar_monotone(iv, tau)
    w = (rep**2) * tau[None, :]
    assert np.all(np.diff(w, axis=1) >= -1e-10)

    report = check_iv_surface_arbitrage(rep, m, tau, spot=100.0, rate=0.0)
    assert report.calendar_ok


def test_butterfly_repair_on_concave_smile() -> None:
    m = np.linspace(0.7, 1.3, 21)
    tau = np.linspace(0.2, 1.0, 8)
    iv = np.empty((m.size, tau.size), dtype=float)
    for j in range(tau.size):
        iv[:, j] = 0.40 - 0.30 * np.exp(-((m - 1.0) ** 2) / 0.01)

    before = check_iv_surface_arbitrage(iv, m, tau, spot=100.0, rate=0.0)
    assert not before.butterfly_ok

    rep = repair_butterfly_convex(iv, m, tau, spot=100.0, rate=0.0)
    after = check_iv_surface_arbitrage(rep, m, tau, spot=100.0, rate=0.0, tol=1e-6)
    assert after.butterfly_ok
    assert after.bounds_ok


def test_volgan_generative_settings_only_repair_when_violated() -> None:
    m = np.linspace(0.7, 1.3, 21)
    tau = np.linspace(0.2, 1.0, 8)
    iv = _flat_bs_surface(m, tau, 0.25)
    settings = volgan_generative_repair_settings(tol=1e-6)
    assert settings.only_if_violated is True
    repaired = repair_iv_surface(iv, m, tau, spot=100.0, rate=0.0, settings=settings)
    assert np.allclose(repaired, iv)


def test_full_repair_pipeline_on_perturbed_flat_surface() -> None:
    m = np.linspace(0.5, 2.5, 31)
    tau = np.linspace(0.05, 2.0, 15)
    rng = np.random.default_rng(42)
    iv = _flat_bs_surface(m, tau, 0.25)
    noise = rng.normal(0.0, 0.02, size=iv.shape)
    iv = np.clip(iv + noise, 0.05, 1.5)
    iv[:, 3] *= 0.85  # inject calendar stress

    settings = SurfaceRepairSettings(
        smooth_sigma_log_moneyness=0.5,
        smooth_sigma_tau=0.3,
        max_iterations=8,
        tol=1e-6,
    )
    repaired = repair_iv_surface(iv, m, tau, spot=100.0, rate=0.02, settings=settings)
    report = check_iv_surface_arbitrage(repaired, m, tau, spot=100.0, rate=0.02, tol=1e-6)
    assert report.arbitrage_free
