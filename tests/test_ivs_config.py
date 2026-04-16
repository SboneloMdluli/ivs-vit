"""Tests for the ``ivs_config`` package."""

from __future__ import annotations

from pathlib import Path

import pytest

from implied_volatility_diffusion.synthetic_ivs_generator.heston_iv_goals import HestonIvGoal
from implied_volatility_diffusion.synthetic_ivs_generator.heston_iv_surface import (
    load_heston_iv_surface_config,
    load_heston_iv_surface_goal_config,
)
from ivs_config import load_config, merge_config, merge_config_files


def test_merge_config_deep() -> None:
    root = Path(__file__).resolve().parents[1]
    base = load_heston_iv_surface_config(root / "config")
    merged = merge_config(
        base,
        {"lhs": {"n_samples": 7}, "market": {"spot": 200.0}},
    )
    assert merged["lhs"]["n_samples"] == 7
    assert merged["lhs"]["seed"] == base["lhs"]["seed"]
    assert merged["market"]["spot"] == 200.0
    assert merged["market"]["dividend_yield"] == base["market"]["dividend_yield"]
    assert base["lhs"]["n_samples"] != 7


def test_heston_goal_configs_merge() -> None:
    root = Path(__file__).resolve().parents[1]
    cfg_dir = root / "config"
    low = load_heston_iv_surface_goal_config(cfg_dir, HestonIvGoal.LOW_VOL)
    assert low["heston_ranges"]["v0"] == [0.0025, 0.04]
    assert low["plot_surface"]["zlim"] == [0.0, 1.0]

    high = load_heston_iv_surface_goal_config(cfg_dir, HestonIvGoal.HIGH_VOL)
    assert high["heston_ranges"]["v0"][1] == 0.45
    assert high["implied_vol"]["sigma_hi"] == 1.0

    skew = load_heston_iv_surface_goal_config(cfg_dir, HestonIvGoal.SKEW)
    assert skew["implied_vol"]["m_extrapolate_below"] == 0.52
    assert skew["grid"]["moneyness"]["end_point"] == 1.6
    assert skew["heston_ranges"]["rho"][0] == -0.99

    smile = load_heston_iv_surface_goal_config(cfg_dir, HestonIvGoal.SMILE)
    assert smile["heston_ranges"]["rho"] == [-0.42, 0.42]
    assert smile["heston_ranges"]["sigma"][0] == 0.6
    assert smile["grid"]["moneyness"]["start_point"] == 0.35

    seq = load_heston_iv_surface_goal_config(cfg_dir, "sequential_path")
    assert seq["sequential_ivs"]["n_steps"] == 32
    assert seq["heston_ranges"]["v0"] == load_heston_iv_surface_config(cfg_dir)["heston_ranges"]["v0"]


def test_load_heston_goal_unknown_raises() -> None:
    root = Path(__file__).resolve().parents[1]
    with pytest.raises(ValueError, match="unknown goal"):
        load_heston_iv_surface_goal_config(root / "config", "not_a_goal")


def test_merge_config_files_order() -> None:
    root = Path(__file__).resolve().parents[1]
    cfg_dir = root / "config"
    merged = merge_config_files(cfg_dir / "heston_iv_surface.yaml", cfg_dir / "iv_surface_grid.yaml")
    assert "heston_ranges" in merged and "grid" in merged and "plot_surface" in merged
    assert merged["grid"]["moneyness"]["start_point"] == 0.5


def test_load_config_empty_file(tmp_path: Path) -> None:
    p = tmp_path / "empty.yaml"
    p.write_text("", encoding="utf-8")
    assert load_config(p) == {}


def test_load_config_rejects_non_mapping_root(tmp_path: Path) -> None:
    p = tmp_path / "bad.yaml"
    p.write_text("- a\n- b\n", encoding="utf-8")
    with pytest.raises(TypeError, match="mapping"):
        load_config(p)
