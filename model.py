"""Mini Transformer encoder implemented from scratch.

Following the assignment brief I may use:

* ``nn.Linear``, ``nn.Embedding``, ``nn.LayerNorm``, ``nn.Dropout``
* general PyTorch tensor / autograd / optimisation primitives

But I may NOT use:

* ``nn.Transformer`` / ``nn.TransformerEncoder`` / ``nn.TransformerEncoderLayer``
* ``nn.MultiheadAttention``
* Hugging Face Transformer classes or pretrained models

Every module below is written out explicitly so the maths is visible
and verifiable rather than hidden behind a single high-level call.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from data import PAD_ID, VOCAB_SIZE, MAX_LEN


# ---------------------------------------------------------------------------
# Positional encoding
# ---------------------------------------------------------------------------
class SinusoidalPositionalEncoding(nn.Module):
    """Fixed sinusoidal positional encoding (Vaswani et al., 2017).

    Stored as a non-trainable buffer of shape ``[1, max_len, d_model]``.
    """

    def __init__(self, d_model: int, max_len: int = MAX_LEN):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D]
        T = x.size(1)
        return x + self.pe[:, :T, :]


# ---------------------------------------------------------------------------
# Scaled dot-product attention and multi-head self-attention
# ---------------------------------------------------------------------------
def scaled_dot_product_attention(
    q: torch.Tensor,           # [B, H, T, d_k]
    k: torch.Tensor,           # [B, H, T, d_k]
    v: torch.Tensor,           # [B, H, T, d_v]
    mask: Optional[torch.Tensor] = None,  # [B, 1, 1, T] with 1=keep, 0=pad
    dropout: Optional[nn.Dropout] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Standard ``softmax(Q K^T / sqrt(d_k)) V`` with additive masking."""
    d_k = q.size(-1)
    # [B, H, T, T]
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        # Where mask == 0 I want -inf so softmax gives 0 probability.
        scores = scores.masked_fill(mask == 0, float("-inf"))

    attn = F.softmax(scores, dim=-1)

    # If a whole row is fully masked (shouldn't happen here but keeps us safe)
    # softmax of all -inf returns NaN -> replace with 0.
    attn = torch.nan_to_num(attn, nan=0.0)

    if dropout is not None:
        attn = dropout(attn)

    out = torch.matmul(attn, v)  # [B, H, T, d_v]
    return out, attn


class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention built from scratch."""

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0, (
            f"d_model={d_model} must be divisible by num_heads={num_heads}"
        )
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads

        # Separate projections for Q, K, V - equivalent to a single linear
        # of size 3*d_model but a bit easier to read.
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.attn_dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,                       # [B, T, D]
        key_padding_mask: Optional[torch.Tensor] = None,  # [B, T] 1=keep,0=pad
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, T, D = x.shape
        H = self.num_heads

        # Project and split into heads -> [B, H, T, d_head]
        q = self.W_q(x).view(B, T, H, self.d_head).transpose(1, 2)
        k = self.W_k(x).view(B, T, H, self.d_head).transpose(1, 2)
        v = self.W_v(x).view(B, T, H, self.d_head).transpose(1, 2)

        # Build attention mask: [B, 1, 1, T]
        mask = None
        if key_padding_mask is not None:
            mask = key_padding_mask.unsqueeze(1).unsqueeze(2)

        out, attn = scaled_dot_product_attention(
            q, k, v, mask=mask, dropout=self.attn_dropout
        )
        # Merge heads: [B, T, D]
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        return self.W_o(out), attn


# ---------------------------------------------------------------------------
# Position-wise FFN and encoder block
# ---------------------------------------------------------------------------
class PositionwiseFeedForward(nn.Module):
    """Two-layer MLP applied independently to each position."""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.dropout(F.relu(self.fc1(x))))


class TransformerEncoderBlock(nn.Module):
    """Pre-LN encoder block:

        h = x + Dropout(MHA(LN(x)))
        y = h + Dropout(FFN(LN(h)))
    """

    def __init__(
        self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1
    ):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.mha = MultiHeadSelfAttention(d_model, num_heads, dropout=dropout)
        self.drop1 = nn.Dropout(dropout)

        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout=dropout)
        self.drop2 = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        attn_out, _ = self.mha(self.ln1(x), key_padding_mask=key_padding_mask)
        x = x + self.drop1(attn_out)

        ffn_out = self.ffn(self.ln2(x))
        x = x + self.drop2(ffn_out)
        return x


# ---------------------------------------------------------------------------
# Full classification model
# ---------------------------------------------------------------------------
class MiniTransformerClassifier(nn.Module):
    """Token embeddings -> (optional) positional encoding -> N encoder blocks
    -> mean pooling over valid tokens -> linear classifier -> 2 logits.
    """

    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE,
        d_model: int = 64,
        num_heads: int = 4,
        d_ff: int = 128,
        num_layers: int = 1,
        dropout: float = 0.1,
        max_len: int = MAX_LEN,
        use_positional_encoding: bool = True,
        num_classes: int = 2,
        pad_id: int = PAD_ID,
    ):
        super().__init__()
        self.pad_id = pad_id
        self.use_positional_encoding = use_positional_encoding

        self.token_emb = nn.Embedding(
            vocab_size, d_model, padding_idx=pad_id
        )
        if use_positional_encoding:
            self.pos_enc = SinusoidalPositionalEncoding(d_model, max_len=max_len)
        else:
            self.pos_enc = None

        self.emb_dropout = nn.Dropout(dropout)

        self.layers = nn.ModuleList(
            [
                TransformerEncoderBlock(d_model, num_heads, d_ff, dropout=dropout)
                for _ in range(num_layers)
            ]
        )
        self.final_ln = nn.LayerNorm(d_model)

        self.classifier = nn.Linear(d_model, num_classes)

    # --- helpers -----------------------------------------------------------
    def _mean_pool(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Mean-pool over non-padding positions.

        x:    [B, T, D]
        mask: [B, T] with 1 = real token, 0 = PAD
        """
        m = mask.unsqueeze(-1).float()     # [B, T, 1]
        summed = (x * m).sum(dim=1)        # [B, D]
        counts = m.sum(dim=1).clamp(min=1.0)  # avoid div-by-zero
        return summed / counts

    # --- forward -----------------------------------------------------------
    def forward(
        self,
        tokens: torch.Tensor,              # [B, T] long
        attention_mask: Optional[torch.Tensor] = None,  # [B, T] 1=keep
    ) -> torch.Tensor:
        if attention_mask is None:
            attention_mask = (tokens != self.pad_id).long()

        x = self.token_emb(tokens)          # [B, T, D]
        if self.pos_enc is not None:
            x = self.pos_enc(x)
        x = self.emb_dropout(x)

        for layer in self.layers:
            x = layer(x, key_padding_mask=attention_mask)

        x = self.final_ln(x)
        pooled = self._mean_pool(x, attention_mask)  # [B, D]
        logits = self.classifier(pooled)             # [B, num_classes]
        return logits


# ---------------------------------------------------------------------------
# Minimal self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    model = MiniTransformerClassifier(
        d_model=64, num_heads=4, d_ff=128, num_layers=1, use_positional_encoding=True
    )
    tokens = torch.tensor([[1, 2, 3, 0, 0], [4, 1, 2, 3, 0]])
    mask = (tokens != 0).long()
    logits = model(tokens, mask)
    print("logits shape:", logits.shape)
    print("#params:", sum(p.numel() for p in model.parameters()))
