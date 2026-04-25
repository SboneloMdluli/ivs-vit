# Unified grid and per-pixel log-IV normalization

This module ensures **pixel consistency** across every surface we generate:
historical SPX quotes, synthetic SABR, and synthetic Heston all share the same
`(k, τ)` cells before any downstream training step sees them.

## 1. The canonical grid

`config/unified_iv_grid.yaml` pins the axes to:

- log-moneyness `k = log(K/S)` in `[-0.5, 0.5]` (step `0.025` → 41 points)
- time to maturity `τ` in `[0.05, 2.0]` years (step `0.05` → 40 points)

The range covers the liquid SPX wing as well as the standard synthetic SABR /
Heston bounds.

```python
from implied_volatility_diffusion import UnifiedGrid

grid = UnifiedGrid.default()           # 41 × 40 canonical axes
grid.shape                             # (41, 40)
grid.log_moneyness, grid.tau, grid.moneyness  # axes + exp(k) helper
```

Alternatively, load straight from YAML:

```python
from pathlib import Path
grid = UnifiedGrid.load(Path("config/unified_iv_grid.yaml"))
```

### Plug into existing synthetic recipes

The SABR / Heston synthetic builders already consume a `grid.moneyness` /
`grid.tau` block through `implied_volatility_diffusion.core.grid.grid_axes`.
`UnifiedGrid.with_config(cfg)` returns a new config dict whose `grid` block
has been replaced by the unified axes, so you get pixel-consistent output
with **zero** change to the generators:

```python
from implied_volatility_diffusion import (
    UnifiedGrid, load_heston_iv_surface_config, implied_vol_surfaces_lhs,
    merge_config_files,
)
from implied_volatility_diffusion.synthetic.sabr import implied_vol_surfaces_sabr_lhs

grid = UnifiedGrid.default()

# Heston
heston_cfg = grid.with_config(load_heston_iv_surface_config("config"))
_, m_h, tau_h, iv_h = implied_vol_surfaces_lhs(heston_cfg)

# SABR (synthetic)
sabr_cfg = grid.with_config(merge_config_files("config/sabr_iv_surface.yaml"))
_, m_s, tau_s, iv_s = implied_vol_surfaces_sabr_lhs(sabr_cfg)

# iv_h[:, i, j] and iv_s[:, i, j] now refer to identical (k_i, tau_j) cells.
```

### Resample anything else onto the grid

For historical surfaces that were built on user-chosen axes (e.g. a
SABR-on-historic-data surface or a kernel-smoothed 60×60 surface), use the
single-shot resampler:

```python
from implied_volatility_diffusion import resample_to_unified_grid

Z_unified = resample_to_unified_grid(
    Z_any,
    k_src=k_grid,         # or m_src= for moneyness source
    tau_src=tau_grid,
    grid=grid,
)
```

Out-of-range pixels are set to `NaN` so downstream code sees the same
masking convention as the kernel smoother.

## 2. Per-pixel log-IV normalization

Given any stack of surfaces `IV` of shape `(N, I, J)` already on the unified
grid, the normalizer fits

```
z_{i,j}^{(n)} = (log σ_{i,j}^{(n)} − μ_{i,j}) / σ_{norm, i,j}
```

so a normalized pixel value of `0.0` always means "historical mean log-IV
for this specific strike/maturity cell", regardless of data source.

```python
from implied_volatility_diffusion import SurfaceNormalizer
import numpy as np

combined = np.concatenate([iv_hist_stack, iv_sabr_stack, iv_heston_stack], axis=0)
norm = SurfaceNormalizer(grid_shape=grid.shape).fit(combined)

z_hist = norm.transform(iv_hist_stack)      # same shape
iv_back = norm.inverse_transform(z_hist)    # exp(z*σ + μ)

norm.save("data/processed/iv_normalizer.npz")
norm2 = SurfaceNormalizer.load("data/processed/iv_normalizer.npz")
```

Features:

- NaN-safe: missing cells are ignored pixel-by-pixel when computing `μ` and `σ_norm`.
- `partial_fit(chunk)` uses a Welford streaming update - handy when the full
  training stack doesn't fit in memory.
- A configurable `sigma_floor` (default `1e-6` in log-IV units) guards against
  division by zero on pixels whose historical log-IV is essentially constant.
- Serialization is plain `.npz`, so inspecting the stats in a notebook is
  as simple as `np.load(path)`.

### Denormalization

The inverse map (diffusion-model output / z-score surface → raw `σ`) is exposed
in three equivalent forms, pick whichever fits the call-site best:

```python
norm.denormalize(z)            # instance method, canonical entry point
norm.inverse_transform(z)      # scikit-learn style alias
denormalize_surface(z, norm.mean, norm.std)   # stateless helper
```

All three implement

```
σ_{i,j}(z) = exp(z_{i,j} · σ_norm_{i,j} + μ_{i,j})
```

and therefore satisfy `denormalize(normalize(σ)) == σ` (up to the `iv_floor` and
`sigma_floor` clamps). In particular, feeding in a zero z-score always recovers
the per-pixel historical mean:

```python
import numpy as np
sigma_mean = norm.denormalize(np.zeros(grid.shape))
assert np.allclose(sigma_mean, np.exp(norm.mean))
```

Pass `return_log_iv=True` when the downstream consumer works directly in
`log σ` space (e.g. further statistical aggregation or arbitrage checks on
total variance).

## 3. End-to-end demo

`notebooks/historical_sabr_interpolation.ipynb` runs through the full pipeline:
build a historical SABR surface, generate synthetic SABR + Heston surfaces on
the same unified grid, fit the normalizer on the combined stack, and confirm
that a surface's z-score at each pixel is comparable across data sources.
