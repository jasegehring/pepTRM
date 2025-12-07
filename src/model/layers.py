"""
Shared model components.
"""

import math
import torch
import torch.nn as nn
from torch import Tensor


class LayerNorm(nn.Module):
    """Layer normalization with optional bias."""

    def __init__(self, dim: int, bias: bool = True, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim)) if bias else None
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        return nn.functional.layer_norm(
            x, self.weight.shape, self.weight, self.bias, self.eps
        )


class FeedForward(nn.Module):
    """Feed-forward network with GELU activation."""

    def __init__(self, dim: int, hidden_dim: int = None, dropout: float = 0.1):
        super().__init__()
        hidden_dim = hidden_dim or dim * 4
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class MultiHeadAttention(nn.Module):
    """Multi-head attention with optional masking."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        causal: bool = False,
    ):
        super().__init__()
        assert dim % num_heads == 0

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.causal = causal

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        q: Tensor,
        k: Tensor = None,
        v: Tensor = None,
        mask: Tensor = None,
    ) -> Tensor:
        """
        Args:
            q: Query tensor (batch, seq_len, dim)
            k: Key tensor (batch, seq_len, dim) - defaults to q (self-attention)
            v: Value tensor (batch, seq_len, dim) - defaults to k
            mask: Attention mask (batch, seq_len) or (batch, q_len, k_len)
        """
        if k is None:
            k = q
        if v is None:
            v = k

        batch, q_len, _ = q.shape
        _, k_len, _ = k.shape

        # Project and reshape to (batch, num_heads, seq_len, head_dim)
        q = self.q_proj(q).view(batch, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(k).view(batch, k_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(v).view(batch, k_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Apply mask
        if mask is not None:
            if mask.dim() == 2:
                # (batch, k_len) -> (batch, 1, 1, k_len)
                mask = mask.unsqueeze(1).unsqueeze(2)
            elif mask.dim() == 3:
                # (batch, q_len, k_len) -> (batch, 1, q_len, k_len)
                mask = mask.unsqueeze(1)
            scores = scores.masked_fill(~mask, float('-inf'))

        # Causal mask
        if self.causal:
            causal_mask = torch.triu(
                torch.ones(q_len, k_len, dtype=torch.bool, device=q.device),
                diagonal=1
            )
            scores = scores.masked_fill(causal_mask, float('-inf'))

        # Softmax and apply to values
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(batch, q_len, self.dim)

        return self.out_proj(out)


class TransformerEncoderLayer(nn.Module):
    """Single transformer encoder layer."""

    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(dim, num_heads, dropout)
        self.ff = FeedForward(dim, dropout=dropout)
        self.norm1 = LayerNorm(dim)
        self.norm2 = LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        # Self-attention with residual
        x = x + self.dropout(self.self_attn(self.norm1(x), mask=mask))
        # Feed-forward with residual
        x = x + self.dropout(self.ff(self.norm2(x)))
        return x


class TransformerDecoderLayer(nn.Module):
    """Single transformer decoder layer with cross-attention."""

    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(dim, num_heads, dropout, causal=False)
        self.cross_attn = MultiHeadAttention(dim, num_heads, dropout)
        self.ff = FeedForward(dim, dropout=dropout)
        self.norm1 = LayerNorm(dim)
        self.norm2 = LayerNorm(dim)
        self.norm3 = LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: Tensor,
        context: Tensor,
        self_mask: Tensor = None,
        context_mask: Tensor = None,
    ) -> Tensor:
        # Self-attention
        x = x + self.dropout(self.self_attn(self.norm1(x), mask=self_mask))
        # Cross-attention to context (encoded spectrum)
        x = x + self.dropout(self.cross_attn(self.norm2(x), context, context, mask=context_mask))
        # Feed-forward
        x = x + self.dropout(self.ff(self.norm3(x)))
        return x
