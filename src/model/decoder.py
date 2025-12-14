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

    Key features (v2 - Step Embeddings & Gated Residuals):
    - Step embeddings: Model knows which refinement iteration it's on (0-7)
      Enables different behavior at different stages (bold early, conservative late)
    - Gated residuals: GRU-style update in answer_step
      new_logits = (1 - gate) * prev_logits + gate * candidate_logits
      Gate starts biased toward 0 (conservative), learns when to update
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        num_layers: int = 2,
        num_heads: int = 4,
        max_seq_len: int = 35,
        vocab_size: int = VOCAB_SIZE,
        num_supervision_steps: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size
        self.num_supervision_steps = num_supervision_steps

        # Token embedding for sequence positions
        self.token_embedding = nn.Embedding(vocab_size, hidden_dim)
        self.position_embedding = nn.Embedding(max_seq_len, hidden_dim)

        # Step embedding: tells model which refinement iteration it's on
        # Critical for learning stage-specific behavior
        self.step_embedding = nn.Embedding(num_supervision_steps, hidden_dim)

        # Project soft sequence representation (probabilities -> embeddings)
        self.soft_embed_proj = nn.Linear(vocab_size, hidden_dim)

        # Shared transformer decoder layers
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(hidden_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])

        self.norm = LayerNorm(hidden_dim)

        # Output head: hidden -> vocab logits (candidate prediction)
        self.output_head = nn.Linear(hidden_dim, vocab_size)

        # Gate head for GRU-style gated residual
        # Outputs per-position, per-vocab gate values in [0, 1]
        # gate=0: keep previous, gate=1: use new candidate
        self.gate_head = nn.Linear(hidden_dim, vocab_size)
        # Initialize gate bias to -2.0 (starts conservative, ~sigmoid(-2)=0.12)
        # Forces model to initially rely on previous step, only update when confident
        nn.init.constant_(self.gate_head.bias, -2.0)

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
        step_idx: int = 0,         # Current supervision step index
    ) -> Tensor:
        """
        Single latent reasoning step: z <- f(x, y, z, t)

        Updates the latent state based on spectrum, current prediction, previous state,
        and the current step index (for step-aware processing).
        """
        batch_size, seq_len = latent_z.shape[:2]
        device = latent_z.device

        # Create input by combining current prediction embedding and latent state
        y_embed = self.soft_embed(current_probs)

        # Add positional information
        positions = torch.arange(seq_len, device=device)
        pos_embed = self.position_embedding(positions)

        # Get step embedding (broadcast to all positions)
        step_idx_tensor = torch.tensor(step_idx, device=device)
        step_emb = self.step_embedding(step_idx_tensor)  # (hidden_dim,)

        # Combine: y_embed + z + position + step
        x = y_embed + latent_z + pos_embed + step_emb

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
        prev_logits: Tensor,    # (batch, seq_len, vocab_size) - previous step's logits
        step_idx: int = 0,      # Current supervision step index
    ) -> Tensor:
        """
        Answer update step with GRU-style gated residual: y <- g(y, z, y_prev, t)

        Uses a learned gate to interpolate between previous prediction and new candidate:
            new_logits = (1 - gate) * prev_logits + gate * candidate_logits

        The gate starts biased toward 0 (keep previous), forcing the model to
        "earn" the right to make changes by learning when updates are beneficial.

        Returns:
            (batch, seq_len, vocab_size) - updated logits
        """
        batch_size, seq_len = latent_z.shape[:2]
        device = latent_z.device

        # 1. Fuse inputs: current prediction embedding + latent state + position + step
        y_embed = self.soft_embed(current_probs)

        positions = torch.arange(seq_len, device=device)
        pos_embed = self.position_embedding(positions)

        step_idx_tensor = torch.tensor(step_idx, device=device)
        step_emb = self.step_embedding(step_idx_tensor)  # (hidden_dim,)

        x = y_embed + latent_z + pos_embed + step_emb

        # 2. Process through transformer layers (self-attention only)
        for layer in self.layers:
            x = layer(x, context=x)

        x = self.norm(x)

        # 3. Generate candidate prediction (what we think the answer should be)
        candidate_logits = self.output_head(x)

        # 4. Generate gate (confidence switch)
        # gate ∈ [0, 1]: 0 = keep previous, 1 = use candidate
        # Bias initialized to -2.0, so initial gate ≈ 0.12 (conservative)
        gate = torch.sigmoid(self.gate_head(x))

        # 5. GRU-style gated update
        new_logits = (1 - gate) * prev_logits + gate * candidate_logits

        return new_logits


class RecursiveCore(nn.Module):
    """
    Complete recursive reasoning core.

    Implements the full TRM recursive loop with step embeddings and gated residuals:
    For each supervision step t:
        For n latent steps: z <- f(x, y, z, t)
        y <- g(y, z, y_prev, t)  # GRU-style gated update

    Key improvements (v2):
    - Step embeddings: Model knows which iteration it's on
    - Gated residuals: Learned interpolation between prev and new predictions
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        num_layers: int = 2,
        num_heads: int = 4,
        max_seq_len: int = 35,
        vocab_size: int = VOCAB_SIZE,
        num_latent_steps: int = 4,
        num_supervision_steps: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.num_latent_steps = num_latent_steps
        self.num_supervision_steps = num_supervision_steps

        self.decoder = RecursiveDecoder(
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            max_seq_len=max_seq_len,
            vocab_size=vocab_size,
            num_supervision_steps=num_supervision_steps,
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
        Full recursive forward pass with step embeddings and gated residuals.

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

            # Store previous logits for gated residual
            prev_logits = y_logits

            # Latent reasoning: n steps of z <- f(x, y, z, t)
            for _ in range(self.num_latent_steps):
                z = self.decoder.latent_step(
                    encoded_spectrum, spectrum_mask, y_probs, z, step_idx=t
                )

            # Answer update with gated residual: y <- g(y, z, y_prev, t)
            y_logits = self.decoder.answer_step(y_probs, z, prev_logits, step_idx=t)

            if return_all_steps:
                all_logits.append(y_logits)

        if return_all_steps:
            return torch.stack(all_logits, dim=0), z  # (T, batch, seq, vocab)
        else:
            return y_logits, z
