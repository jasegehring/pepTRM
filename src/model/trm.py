"""
Complete Tiny Recursive Model for peptide sequencing.
"""

import torch
import torch.nn as nn
from torch import Tensor
from dataclasses import dataclass
from typing import Optional

from .encoder import SpectrumEncoder
from .decoder import RecursiveCore
from ..constants import VOCAB_SIZE


@dataclass
class TRMConfig:
    """Model configuration."""
    hidden_dim: int = 384  # INCREASED from 256 (1.5x capacity)
    num_encoder_layers: int = 3  # INCREASED from 2
    num_decoder_layers: int = 3  # INCREASED from 2
    num_heads: int = 6  # INCREASED from 4 (384/6 = 64 per head)
    max_peaks: int = 100
    max_seq_len: int = 35  # Supports peptides up to 30aa
    max_mass: float = 2000.0
    vocab_size: int = VOCAB_SIZE
    num_supervision_steps: int = 8
    num_latent_steps: int = 6  # INCREASED from 4 (original TRM value)
    dropout: float = 0.1


class RecursivePeptideModel(nn.Module):
    """
    Tiny Recursive Model for de novo peptide sequencing.

    Architecture:
        1. Spectrum Encoder: Transformer encoder on peak embeddings
        2. Recursive Core: Shared weights, runs T supervision steps
        3. Output: Sequence logits at each step (for deep supervision)

    Key features:
        - Uses both mass AND intensity from spectrum peaks
        - Includes precursor mass as explicit input
        - Soft embeddings from probability distributions
        - Deep supervision at every step
    """

    def __init__(self, config: TRMConfig):
        super().__init__()
        self.config = config

        # Spectrum encoder
        self.encoder = SpectrumEncoder(
            hidden_dim=config.hidden_dim,
            num_layers=config.num_encoder_layers,
            num_heads=config.num_heads,
            max_peaks=config.max_peaks,
            max_mass=config.max_mass,
            dropout=config.dropout,
        )

        # Recursive decoder core
        self.recursive_core = RecursiveCore(
            hidden_dim=config.hidden_dim,
            num_layers=config.num_decoder_layers,
            num_heads=config.num_heads,
            max_seq_len=config.max_seq_len,
            vocab_size=config.vocab_size,
            num_latent_steps=config.num_latent_steps,
            dropout=config.dropout,
        )

    def forward(
        self,
        spectrum_masses: Tensor,
        spectrum_intensities: Tensor,
        spectrum_mask: Tensor,
        precursor_mass: Tensor,
        precursor_charge: Tensor,
        num_supervision_steps: Optional[int] = None,
    ) -> tuple[Tensor, Tensor]:
        """
        Forward pass.

        Args:
            spectrum_masses: (batch, max_peaks) - peak m/z values
            spectrum_intensities: (batch, max_peaks) - peak intensities
            spectrum_mask: (batch, max_peaks) - True for real peaks
            precursor_mass: (batch,) - precursor mass
            precursor_charge: (batch,) - precursor charge state
            num_supervision_steps: Override default number of steps

        Returns:
            all_logits: (T, batch, max_seq_len, vocab_size) - logits at each step
            final_z: (batch, max_seq_len, hidden_dim) - final latent state
        """
        num_steps = num_supervision_steps or self.config.num_supervision_steps

        # Encode spectrum
        encoded_spectrum, full_mask = self.encoder(
            spectrum_masses,
            spectrum_intensities,
            spectrum_mask,
            precursor_mass,
            precursor_charge,
        )

        # Recursive refinement
        all_logits, final_z = self.recursive_core(
            encoded_spectrum,
            full_mask,
            num_supervision_steps=num_steps,
            return_all_steps=True,
        )

        return all_logits, final_z

    def predict(
        self,
        spectrum_masses: Tensor,
        spectrum_intensities: Tensor,
        spectrum_mask: Tensor,
        precursor_mass: Tensor,
        precursor_charge: Tensor,
    ) -> Tensor:
        """
        Inference: return only final predictions.

        Returns:
            logits: (batch, max_seq_len, vocab_size)
        """
        all_logits, _ = self.forward(
            spectrum_masses,
            spectrum_intensities,
            spectrum_mask,
            precursor_mass,
            precursor_charge,
        )
        return all_logits[-1]  # Return final step


def create_model(config: TRMConfig = None) -> RecursivePeptideModel:
    """Factory function to create model."""
    config = config or TRMConfig()
    return RecursivePeptideModel(config)
