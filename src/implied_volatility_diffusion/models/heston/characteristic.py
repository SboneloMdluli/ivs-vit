"""Risk-neutral Heston characteristic function (PyTorch complex-valued)."""

from __future__ import annotations

import torch


def heston_cf(
    u: torch.Tensor,
    tau: float | torch.Tensor,
    spot: float | torch.Tensor,
    rate: float | torch.Tensor,
    dividend_yield: float | torch.Tensor,
    kappa: float | torch.Tensor,
    theta: float | torch.Tensor,
    sigma: float | torch.Tensor,
    rho: float | torch.Tensor,
    v0: float | torch.Tensor,
) -> torch.Tensor:
    """Characteristic function of ``ln(S_T)`` under the risk-neutral Heston model.

    ``u`` is a real tensor of Fourier frequencies; the rest are scalars or
    tensors broadcastable with ``u``. The return tensor is ``complex128`` and
    has the same shape as ``u`` (post-broadcast).

    Reference: Gatheral (2011), *The Volatility Surface*; Fang & Oosterlee.
    """
    if not isinstance(u, torch.Tensor):
        raise TypeError("u must be a torch.Tensor")

    cdtype = torch.complex128 if u.dtype == torch.float64 else torch.complex64
    u_c = u.to(cdtype)

    def _c(x):
        return torch.as_tensor(x, dtype=cdtype, device=u.device)

    tau_c = _c(tau)
    spot_c = _c(spot)
    r_c = _c(rate)
    q_c = _c(dividend_yield)
    kappa_c = _c(kappa)
    theta_c = _c(theta)
    sigma_c = _c(sigma)
    rho_c = _c(rho)
    v0_c = _c(v0)

    iu = 1j * u_c
    rmq = r_c - q_c
    sig2 = sigma_c * sigma_c

    p = kappa_c - rho_c * sigma_c * iu
    q_term = sig2 * (iu + u_c * u_c)
    d = torch.sqrt(p * p + q_term)
    # Numerical guards matching the NumPy reference.
    d = torch.where(d.abs() < 1e-14, torch.as_tensor(1e-14 + 0j, dtype=cdtype, device=u.device), d)

    exp_mdt = torch.exp(-d * tau_c)
    g = (p - d) / (p + d)
    g = torch.where(g.abs() > 1e-12, g, torch.as_tensor(1e-12 + 0j, dtype=cdtype, device=u.device))

    intr = 1.0 - g * exp_mdt
    denom = 1.0 - g

    big_c = rmq * iu * tau_c + (kappa_c * theta_c / sig2) * (
        (p - d) * tau_c - 2.0 * (torch.log(intr) - torch.log(denom))
    )
    big_d = ((p - d) / sig2) * ((1.0 - exp_mdt) / intr)
    return torch.exp(big_c + big_d * v0_c + iu * torch.log(spot_c))
