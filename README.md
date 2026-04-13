# Implied-Volatility-Diffusion-Transformer

## Setup

With [uv](https://github.com/astral-sh/uv):

```bash
uv sync
uv run pytest
```

For Jupyter:

```bash
uv sync --group notebooks
```

## Tests

```bash
uv run pytest
```

## Heston-COS synthetic Implied-Volatility Generator

**Goal:** produce many diverse **Black–Scholes implied volatility** surfaces that are consistent with the **Heston stochastic volatility** model, for training or benchmarking downstream models.

**Parameter sampling** — `lhs_heston_params` / `lhs_heston_params_multi_batch` sample rows in fixed order `HESTON_PARAM_ORDER`. Ranges come from `heston_ranges` in config; optional `lhs.log_uniform` maps selected positive parameters through log-uniform margins for better coverage of scales.

**Pricing** — For each `(strike, τ)` on the grid, `heston_call_cos` (Fang & Oosterlee Fourier–cosine method) evaluates the **discounted European call** under Heston. The implementation uses the Heston characteristic function, cumulant-based truncation of the log-spot interval `[a, b]`, and strike-aware widening so short maturities still bracket `log(K)` reliably.

**Implied volatility** — The model call price is inverted to Black–Scholes σ with `implied_volatility` in `black_scholes.py`: Jäckel’s rational guess via `py_lets_be_rational`, a short Newton refinement on the price residual, and Brent fallback when vega is tiny or the wing is difficult.

**Batch surfaces** — `implied_vol_surfaces_lhs(cfg)` returns `(params, m, tau, iv)` with `iv` of shape `(n_samples, n_moneyness, n_tau)` (total samples = `n_samples × n_batches` when multi-batch LHS is enabled). `implied_vol_surface_for_params` computes a single surface for one parameter vector.

