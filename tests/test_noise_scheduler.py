"""Tests for VP forward noising scheduler APIs."""

import importlib.util
from pathlib import Path

import numpy as np
import torch

_MODULE_PATH = Path(__file__).resolve().parents[1] / "src" / "implied_volatility_diffusion" / "noise_scheduler.py"
_SPEC = importlib.util.spec_from_file_location("noise_scheduler", _MODULE_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError(f"Unable to import noise scheduler from {_MODULE_PATH}")
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
VPNoiseScheduler = _MODULE.VPNoiseScheduler


def test_vp_scheduler_converges_to_standard_normal() -> None:
    """At large t, mean approaches 0 and variance approaches 1."""
    rng = np.random.default_rng(7)
    x0 = np.full(200_000, 3.0, dtype=float)
    beta = 1.0
    t = 20.0

    scheduler = VPNoiseScheduler(beta=beta)
    xt = scheduler.add_noise(x0, t=t, rng=rng)

    mean_emp = float(np.mean(xt))
    var_emp = float(np.var(xt))

    decay = np.exp(-0.5 * beta * t)
    mean_theory = decay * 3.0
    var_theory = 1.0 - np.exp(-beta * t)

    assert np.isclose(mean_emp, mean_theory, atol=1e-2)
    assert np.isclose(var_emp, var_theory, atol=2e-2)


def test_vp_scheduler_variance_follows_closed_form_and_is_bounded() -> None:
    """Variance should follow closed form and remain bounded."""
    rng = np.random.default_rng(11)
    x0 = rng.normal(loc=0.0, scale=2.5, size=200_000)
    var0 = float(np.var(x0))

    beta = 0.8
    scheduler = VPNoiseScheduler(beta=beta)

    for t in [1.0, 5.0, 25.0, 100.0]:
        xt = scheduler.add_noise(x0, t=t, rng=rng)
        var_emp = float(np.var(xt))
        var_theory = np.exp(-beta * t) * var0 + (1.0 - np.exp(-beta * t))
        assert np.isclose(var_emp, var_theory, atol=2e-2)
        assert var_emp <= max(var0, 1.0) + 1e-6


def test_vp_scheduler_add_noise_accepts_2d_image_shape() -> None:
    scheduler = VPNoiseScheduler(beta=1.0)
    image = np.linspace(0.0, 1.0, 64, dtype=float).reshape(8, 8)
    noised = scheduler.add_noise(image, t=0.8, rng=np.random.default_rng(123))
    assert noised.shape == image.shape
    assert np.isfinite(noised).all()


def test_vp_scheduler_q_sample_matches_expected_shape() -> None:
    scheduler = VPNoiseScheduler(timesteps=50, beta_schedule="linear")
    x0 = torch.zeros(4, 1, 8, 8, dtype=torch.float32)
    t = torch.tensor([0, 10, 20, 49], dtype=torch.long)
    xt = scheduler.q_sample(x0, t)
    assert xt.shape == x0.shape
    assert torch.isfinite(xt).all()


def test_forward_process_returns_progression_stack() -> None:
    scheduler = VPNoiseScheduler(timesteps=100)
    x0 = torch.ones(1, 1, 16, 16, dtype=torch.float32)
    t_values = [0, 10, 30, 60, 99]
    progression = scheduler.forward_process(x0, t_values)
    assert progression.shape == (len(t_values),) + x0.shape
    assert torch.isfinite(progression).all()


def test_add_noise_supports_torch_tensor_api() -> None:
    scheduler = VPNoiseScheduler(beta=0.9)
    x0 = torch.zeros(2, 1, 8, 8, dtype=torch.float32)
    noised = scheduler.add_noise(x0, t=1.5, generator=torch.Generator().manual_seed(9))
    assert isinstance(noised, torch.Tensor)
    assert noised.shape == x0.shape
    assert torch.isfinite(noised).all()


def test_forward_process_continuous_mode() -> None:
    scheduler = VPNoiseScheduler(beta=1.0)
    x0 = torch.ones(1, 1, 6, 6, dtype=torch.float32)
    t_values = [0.0, 0.5, 2.0]
    out = scheduler.forward_process(x0, t_values, mode="continuous")
    assert out.shape == (len(t_values),) + x0.shape
    assert torch.isfinite(out).all()
