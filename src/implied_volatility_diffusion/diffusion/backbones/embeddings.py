"""Time/position embeddings shared by U-Net and ViT denoisers."""

import math

import torch
import torch.nn as nn


class SinusoidalTimeEmbedding(nn.Module):
    """Transformer-style sinusoidal embedding for diffusion timesteps."""

    def __init__(self, dim: int) -> None:
        """Create sinusoidal frequencies for an even embedding dimension."""
        super().__init__()
        if dim <= 0 or dim % 2 != 0:
            raise ValueError("SinusoidalTimeEmbedding dim must be a positive even integer")

        self.dim = int(dim)
        # freq_i = exp(-log(10000) * i / (half - 1)),  i = 0..half-1.
        half = dim // 2
        denom = max(half - 1, 1)
        freqs = torch.exp(-math.log(10000.0) * torch.arange(half, dtype=torch.float32) / denom)
        self.register_buffer("freqs", freqs)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Encode timesteps into sinusoidal features."""
        args = t.float().unsqueeze(1) * self.freqs.unsqueeze(0)
        # [sin(t * freq), cos(t * freq)]
        return torch.cat([args.sin(), args.cos()], dim=-1)


class TimeEmbeddingMLP(nn.Module):
    """Standard sinusoidal -> MLP embedding head used by both backbones."""

    def __init__(self, dim: int, hidden_dim: int | None = None) -> None:
        """Build sinusoidal embedding followed by a two-layer MLP."""
        super().__init__()
        hidden_dim = hidden_dim or dim
        self.sinusoidal = SinusoidalTimeEmbedding(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    @property
    def out_dim(self) -> int:
        """Return the embedding width produced by the MLP head."""
        # The hidden_dim is captured in the last Linear layer's out_features.
        return self.mlp[-1].out_features

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Return learned timestep embeddings for the provided indices."""
        return self.mlp(self.sinusoidal(t))


__all__ = ["SinusoidalTimeEmbedding", "TimeEmbeddingMLP"]
