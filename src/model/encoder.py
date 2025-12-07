"""
Spectrum encoder: Transforms MS/MS spectrum into contextual representations.
"""

import torch
import torch.nn as nn
from torch import Tensor

from .layers import TransformerEncoderLayer, LayerNorm
from ..data.encoding import PeakEncoder, PrecursorEncoder


class SpectrumEncoder(nn.Module):
    """
    Encode MS/MS spectrum (list of peaks) into contextual representations.

    Architecture:
    1. Encode each peak (mass, intensity) independently
    2. Add precursor information
    3. Apply Transformer encoder layers
    4. Output: contextualized peak embeddings
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        num_layers: int = 2,
        num_heads: int = 4,
        max_peaks: int = 100,
        max_mass: float = 2000.0,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim

        # Peak encoding (mass + intensity -> hidden_dim)
        self.peak_encoder = PeakEncoder(
            dim=hidden_dim,
            max_mass=max_mass,
            intensity_dim=32,
        )

        # Precursor encoding
        self.precursor_encoder = PrecursorEncoder(
            dim=hidden_dim,
            max_mass=max_mass * 1.5,  # Precursor can be larger
        )

        # Learnable [CLS]-like token for precursor info
        self.precursor_token = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)

        # Transformer encoder layers
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(hidden_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])

        self.norm = LayerNorm(hidden_dim)

    def forward(
        self,
        masses: Tensor,           # (batch, max_peaks)
        intensities: Tensor,      # (batch, max_peaks)
        peak_mask: Tensor,        # (batch, max_peaks)
        precursor_mass: Tensor,   # (batch,)
        precursor_charge: Tensor, # (batch,)
    ) -> tuple[Tensor, Tensor]:
        """
        Encode spectrum.

        Returns:
            encoded: (batch, max_peaks + 1, hidden_dim) - peak representations + precursor
            mask: (batch, max_peaks + 1) - attention mask
        """
        batch_size = masses.shape[0]

        # Encode peaks
        peak_embeddings = self.peak_encoder(masses, intensities)  # (batch, max_peaks, hidden)

        # Encode precursor and add as first token
        precursor_emb = self.precursor_encoder(precursor_mass, precursor_charge)  # (batch, hidden)
        precursor_token = self.precursor_token.expand(batch_size, -1, -1)  # (batch, 1, hidden)
        precursor_token = precursor_token + precursor_emb.unsqueeze(1)

        # Concatenate: [precursor, peaks...]
        x = torch.cat([precursor_token, peak_embeddings], dim=1)  # (batch, 1 + max_peaks, hidden)

        # Update mask to include precursor token
        full_mask = torch.cat([
            torch.ones(batch_size, 1, dtype=torch.bool, device=peak_mask.device),
            peak_mask
        ], dim=1)

        # Apply transformer layers
        for layer in self.layers:
            x = layer(x, mask=full_mask)

        x = self.norm(x)

        return x, full_mask
