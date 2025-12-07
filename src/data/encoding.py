"""
Sinusoidal embeddings for continuous mass values.

Similar to positional encoding in Transformers, but the input is mass (float)
instead of position (integer). This allows the model to learn mass relationships.
"""

import math
import torch
import torch.nn as nn
from torch import Tensor


class SinusoidalMassEmbedding(nn.Module):
    """
    Project scalar mass values to high-dimensional space using sinusoidal functions.

    Uses multiple frequency bands to capture both coarse mass differences
    (hundreds of Da) and fine differences (fractions of Da).

    For a mass value m, the embedding is:
        [sin(m * f_0), cos(m * f_0), sin(m * f_1), cos(m * f_1), ...]

    where frequencies f_i are logarithmically spaced.
    """

    def __init__(
        self,
        dim: int = 256,
        max_mass: float = 2000.0,
        min_freq: float = 1e-4,
        max_freq: float = 1.0,
        learnable_scale: bool = True,
    ):
        """
        Args:
            dim: Output embedding dimension (must be even)
            max_mass: Maximum expected mass value (for normalization)
            min_freq: Minimum frequency for sinusoidal functions
            max_freq: Maximum frequency for sinusoidal functions
            learnable_scale: Whether to include learnable scaling
        """
        super().__init__()
        assert dim % 2 == 0, "Dimension must be even for sin/cos pairs"

        self.dim = dim
        self.max_mass = max_mass

        # Create logarithmically spaced frequencies
        num_freqs = dim // 2
        freqs = torch.exp(
            torch.linspace(
                math.log(min_freq),
                math.log(max_freq),
                num_freqs
            )
        )
        self.register_buffer('freqs', freqs)

        # Optional learnable scaling
        if learnable_scale:
            self.scale = nn.Parameter(torch.ones(dim))
        else:
            self.register_buffer('scale', torch.ones(dim))

    def forward(self, mass: Tensor) -> Tensor:
        """
        Args:
            mass: Tensor of any shape containing mass values

        Returns:
            Tensor of shape (*mass.shape, dim) with embeddings
        """
        # Normalize mass to reasonable range
        mass_normalized = mass / self.max_mass

        # Expand for broadcasting: (..., 1) @ (num_freqs,) -> (..., num_freqs)
        mass_expanded = mass_normalized.unsqueeze(-1)
        angles = mass_expanded * self.freqs * 2 * math.pi

        # Concatenate sin and cos
        embeddings = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)

        return embeddings * self.scale


class PeakEncoder(nn.Module):
    """
    Encode a single spectrum peak (mass, intensity) pair.

    Combines sinusoidal mass embedding with intensity information.
    """

    def __init__(
        self,
        dim: int = 256,
        max_mass: float = 2000.0,
        intensity_dim: int = 32,
    ):
        super().__init__()

        self.mass_embedding = SinusoidalMassEmbedding(
            dim=dim - intensity_dim,
            max_mass=max_mass,
        )

        # Simple MLP for intensity (log-scaled)
        self.intensity_encoder = nn.Sequential(
            nn.Linear(1, intensity_dim),
            nn.ReLU(),
            nn.Linear(intensity_dim, intensity_dim),
        )

        self.output_dim = dim

    def forward(self, mass: Tensor, intensity: Tensor) -> Tensor:
        """
        Args:
            mass: (batch, num_peaks) mass values
            intensity: (batch, num_peaks) intensity values

        Returns:
            (batch, num_peaks, dim) peak embeddings
        """
        # Log-scale intensity (add small epsilon to avoid log(0))
        log_intensity = torch.log(intensity + 1e-8).unsqueeze(-1)

        mass_emb = self.mass_embedding(mass)  # (batch, num_peaks, dim - intensity_dim)
        intensity_emb = self.intensity_encoder(log_intensity)  # (batch, num_peaks, intensity_dim)

        return torch.cat([mass_emb, intensity_emb], dim=-1)


class PrecursorEncoder(nn.Module):
    """
    Encode precursor information (mass and charge).
    """

    def __init__(self, dim: int = 256, max_mass: float = 3000.0, max_charge: int = 6):
        super().__init__()

        self.mass_embedding = SinusoidalMassEmbedding(dim=dim - 16, max_mass=max_mass)
        self.charge_embedding = nn.Embedding(max_charge + 1, 16)

    def forward(self, precursor_mass: Tensor, charge: Tensor) -> Tensor:
        """
        Args:
            precursor_mass: (batch,) precursor masses
            charge: (batch,) charge states (integers)

        Returns:
            (batch, dim) precursor embeddings
        """
        mass_emb = self.mass_embedding(precursor_mass)
        charge_emb = self.charge_embedding(charge)
        return torch.cat([mass_emb, charge_emb], dim=-1)
