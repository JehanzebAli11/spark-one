"""Tiny decoder-only transformer over bytes + image patches, with a KV cache so a shared
prefix (image + state) is computed once and reused across questions."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .tokenizer import IMG, VOCAB


@dataclass
class TinyConfig:
    d_model: int = 128
    n_layers: int = 4
    n_heads: int = 4
    max_len: int = 768
    img_size: int = 64
    patch: int = 8
    dropout: float = 0.0

    @property
    def n_patches(self) -> int:
        return (self.img_size // self.patch) ** 2

    def to_dict(self):
        return asdict(self)


class Block(nn.Module):
    def __init__(self, c: TinyConfig):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(c.d_model), nn.LayerNorm(c.d_model)
        self.qkv = nn.Linear(c.d_model, 3 * c.d_model)
        self.proj = nn.Linear(c.d_model, c.d_model)
        self.mlp = nn.Sequential(nn.Linear(c.d_model, 4 * c.d_model), nn.GELU(),
                                 nn.Linear(4 * c.d_model, c.d_model))
        self.h = c.n_heads
        self.drop = nn.Dropout(c.dropout)

    def forward(self, x, past=None):
        B, T, D = x.shape
        q, k, v = self.qkv(self.ln1(x)).split(D, dim=-1)
        sh = lambda t: t.view(B, T, self.h, D // self.h).transpose(1, 2)
        q, k, v = sh(q), sh(k), sh(v)
        if past is not None:
            pk, pv = past
            k, v = torch.cat([pk, k], 2), torch.cat([pv, v], 2)
        P = k.shape[2] - T
        if P == 0:
            a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:  # queries attend to the whole cached prefix plus causally within themselves
            mask = torch.ones(T, P + T, dtype=torch.bool, device=x.device).tril(diagonal=P)
            a = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        a = a.transpose(1, 2).reshape(B, T, D)
        x = x + self.drop(self.proj(a))
        x = x + self.drop(self.mlp(self.ln2(x)))
        return x, (k, v)


class TinySystemOne(nn.Module):
    def __init__(self, c: TinyConfig):
        super().__init__()
        self.c = c
        self.tok = nn.Embedding(VOCAB, c.d_model)
        self.pos = nn.Embedding(c.max_len, c.d_model)
        self.patch = nn.Linear(3 * c.patch * c.patch, c.d_model)
        self.blocks = nn.ModuleList(Block(c) for _ in range(c.n_layers))
        self.ln = nn.LayerNorm(c.d_model)
        self.head = nn.Linear(c.d_model, VOCAB, bias=False)
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)
            if getattr(m, "bias", None) is not None:
                nn.init.zeros_(m.bias)

    def patchify(self, images: torch.Tensor) -> torch.Tensor:
        """images: [N, 3, S, S] float in [0,1] -> [N, n_patches, d]"""
        p = self.c.patch
        N = images.shape[0]
        x = images.unfold(2, p, p).unfold(3, p, p)  # N,3,h,w,p,p
        x = x.permute(0, 2, 3, 1, 4, 5).reshape(N, -1, 3 * p * p)
        return self.patch(x - 0.5)

    def embed(self, ids: torch.Tensor, images: torch.Tensor | None, offset: int = 0):
        x = self.tok(ids)
        img_mask = ids == IMG
        if img_mask.any():
            has = img_mask.any(1)
            pe = self.patchify(images[has])  # images aligned with batch rows
            x = x.clone()
            x[img_mask] = pe.reshape(-1, x.shape[-1]).to(x.dtype)
        pos = torch.arange(offset, offset + ids.shape[1], device=ids.device)
        return x + self.pos(pos)[None]

    def forward(self, ids, images=None, past=None, offset: int = 0):
        x = self.embed(ids, images, offset)
        new = []
        for i, b in enumerate(self.blocks):
            x, kv = b(x, None if past is None else past[i])
            new.append(kv)
        return self.head(self.ln(x)), new

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
