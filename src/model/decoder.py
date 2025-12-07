"""
Recursive decoder: The core TRM component that iteratively refines predictions.
"""

import torch
import torch.nn as nn
from torch import Tensor

from .layers import TransformerDecoderLayer, LayerNorm
from ..constants import VOCAB_SIZE


class RecursiveDecoder(nn.Module):
    """
    Recursive decoder that iteratively refines sequence predictions.

    Following TRM, this uses a single shared network that:
    1. Updates latent state z based on (x, y, z) - "think"
    2. Updates prediction y based on (y, z) - "act"

    The same network is used for both operations (distinguished by input).
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        num_layers: int = 2,
        num_heads: int = 4,
        max_seq_len: int = 25,
        vocab_size: int = VOCAB_SIZE,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size

        # Token embedding for sequence positions
        self.token_embedding = nn.Embedding(vocab_size, hidden_dim)
        self.position_embedding = nn.Embedding(max_seq_len, hidden_dim)

        # Project soft sequence representation (probabilities -> embeddings)
        self.soft_embed_proj = nn.Linear(vocab_size, hidden_dim)

        # Shared transformer decoder layers
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(hidden_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])

        self.norm = LayerNorm(hidden_dim)

        # Output head: hidden -> vocab logits
        self.output_head = nn.Linear(hidden_dim, vocab_size)

        # Learnable initial latent state
        self.initial_z = nn.Parameter(torch.randn(1, max_seq_len, hidden_dim) * 0.02)

        # Learnable initial sequence guess (uniform-ish)
        self.initial_logits = nn.Parameter(torch.zeros(1, max_seq_len, vocab_size))

    def get_initial_state(self, batch_size: int, device: torch.device) -> tuple[Tensor, Tensor]:
        """
        Get initial prediction (y) and latent state (z).

        Returns:
            y: (batch, max_seq_len, vocab_size) - initial logits
            z: (batch, max_seq_len, hidden_dim) - initial latent state
        """
        y = self.initial_logits.expand(batch_size, -1, -1).clone()
        z = self.initial_z.expand(batch_size, -1, -1).clone()
        return y, z

    def soft_embed(self, probs: Tensor) -> Tensor:
        """
        Convert probability distribution to soft embedding.

        Args:
            probs: (batch, seq_len, vocab_size) - probability distribution

        Returns:
            (batch, seq_len, hidden_dim) - soft embeddings
        """
        return self.soft_embed_proj(probs)

    def latent_step(
        self,
        encoded_spectrum: Tensor,  # (batch, num_peaks, hidden)
        spectrum_mask: Tensor,     # (batch, num_peaks)
        current_probs: Tensor,     # (batch, seq_len, vocab_size)
        latent_z: Tensor,          # (batch, seq_len, hidden)
    ) -> Tensor:
        """
        Single latent reasoning step: z <- f(x, y, z)

        Updates the latent state based on spectrum, current prediction, and previous state.
        """
        batch_size, seq_len = latent_z.shape[:2]

        # Create input by combining current prediction embedding and latent state
        y_embed = self.soft_embed(current_probs)

        # Add positional information
        positions = torch.arange(seq_len, device=latent_z.device)
        pos_embed = self.position_embedding(positions)

        # Combine: y_embed + z + position
        x = y_embed + latent_z + pos_embed

        # Apply decoder layers with cross-attention to spectrum
        for layer in self.layers:
            x = layer(
                x,
                context=encoded_spectrum,
                context_mask=spectrum_mask,
            )

        # Return updated latent state (residual connection)
        return self.norm(x) + latent_z

    def answer_step(
        self,
        current_probs: Tensor,  # (batch, seq_len, vocab_size)
        latent_z: Tensor,       # (batch, seq_len, hidden)
    ) -> Tensor:
        """
        Answer update step: y <- g(y, z)

        Updates the prediction based on current prediction and latent state.

        Returns:
            (batch, seq_len, vocab_size) - updated logits
        """
        batch_size, seq_len = latent_z.shape[:2]

        # Combine current prediction and latent state
        y_embed = self.soft_embed(current_probs)

        positions = torch.arange(seq_len, device=latent_z.device)
        pos_embed = self.position_embedding(positions)

        x = y_embed + latent_z + pos_embed

        # Self-attention only (no cross-attention for answer step)
        for layer in self.layers:
            # Use self-attention only by passing x as both input and context
            x = layer(x, context=x)

        x = self.norm(x)

        # Project to logits
        return self.output_head(x)


class RecursiveCore(nn.Module):
    """
    Complete recursive reasoning core.

    Implements the full TRM recursive loop:
    For each supervision step:
        For n latent steps: z <- f(x, y, z)
        y <- g(y, z)
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        num_layers: int = 2,
        num_heads: int = 4,
        max_seq_len: int = 25,
        vocab_size: int = VOCAB_SIZE,
        num_latent_steps: int = 4,  # MVP: 4 instead of 6
        dropout: float = 0.1,
    ):
        super().__init__()

        self.num_latent_steps = num_latent_steps

        self.decoder = RecursiveDecoder(
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            max_seq_len=max_seq_len,
            vocab_size=vocab_size,
            dropout=dropout,
        )

    def forward(
        self,
        encoded_spectrum: Tensor,
        spectrum_mask: Tensor,
        num_supervision_steps: int = 8,
        return_all_steps: bool = True,
    ) -> tuple[Tensor, Tensor]:
        """
        Full recursive forward pass.

        Args:
            encoded_spectrum: (batch, num_peaks, hidden) - encoded spectrum
            spectrum_mask: (batch, num_peaks) - attention mask
            num_supervision_steps: Number of supervision steps (T in paper)
            return_all_steps: If True, return logits from all steps

        Returns:
            all_logits: (T, batch, seq_len, vocab) if return_all_steps else (batch, seq_len, vocab)
            final_z: (batch, seq_len, hidden) - final latent state
        """
        batch_size = encoded_spectrum.shape[0]
        device = encoded_spectrum.device

        # Initialize
        y_logits, z = self.decoder.get_initial_state(batch_size, device)

        all_logits = []

        for t in range(num_supervision_steps):
            # Convert logits to probabilities for soft embedding
            y_probs = torch.softmax(y_logits, dim=-1)

            # Latent reasoning: n steps of z <- f(x, y, z)
            for _ in range(self.num_latent_steps):
                z = self.decoder.latent_step(
                    encoded_spectrum, spectrum_mask, y_probs, z
                )

            # Answer update: y <- g(y, z)
            y_logits = self.decoder.answer_step(y_probs, z)

            if return_all_steps:
                all_logits.append(y_logits)

        if return_all_steps:
            return torch.stack(all_logits, dim=0), z  # (T, batch, seq, vocab)
        else:
            return y_logits, z
