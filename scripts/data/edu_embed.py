#!/usr/bin/env python3
"""Frozen multilingual sentence-embedding for the edu-quality classifier.

Shared by:
  - train_edu_classifier.py  (embed the labeled docs)
  - score_corpus_edu.py      (embed the full corpus for filtering)

Uses transformers directly (no sentence-transformers dependency): mean-pooling
over the last hidden state with the attention mask, then L2-normalisation.
Defaults to intfloat/multilingual-e5-large, which expects a 'passage: ' prefix
for documents. The SAME embedder config must be used at train and score time —
the trained artifact stores emb_model / prefix / max_length so they stay in sync.
"""
from __future__ import annotations

from typing import List, Optional

import torch

DEFAULT_MODEL = "intfloat/multilingual-e5-large"
DEFAULT_PREFIX = "passage: "


class EduEmbedder:
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        prefix: str = DEFAULT_PREFIX,
        device: Optional[str] = None,
        max_length: int = 512,
        dtype: Optional[torch.dtype] = None,
    ):
        from transformers import AutoModel, AutoTokenizer

        self.model_name = model_name
        self.prefix = prefix
        self.max_length = max_length
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        if dtype is None:
            dtype = torch.float16 if str(device).startswith("cuda") else torch.float32
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name, torch_dtype=dtype)
        self.model.to(device).eval()
        self.dim = int(self.model.config.hidden_size)

    @staticmethod
    def _mean_pool(last_hidden: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        mask = attn_mask.unsqueeze(-1).to(last_hidden.dtype)
        summed = (last_hidden * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        return summed / counts

    @torch.no_grad()
    def embed(self, texts: List[str], batch_size: int = 32, progress_every: int = 0) -> torch.Tensor:
        """Return an (N, dim) float32 CPU tensor of L2-normalised embeddings."""
        out = []
        n = len(texts)
        for i in range(0, n, batch_size):
            batch = [self.prefix + (t or "") for t in texts[i : i + batch_size]]
            enc = self.tok(
                batch, padding=True, truncation=True,
                max_length=self.max_length, return_tensors="pt",
            ).to(self.device)
            res = self.model(**enc)
            emb = self._mean_pool(res.last_hidden_state, enc["attention_mask"])
            emb = torch.nn.functional.normalize(emb, p=2, dim=1)
            out.append(emb.float().cpu())
            if progress_every and (i // batch_size) % progress_every == 0:
                print(f"  embedded {min(i + batch_size, n)}/{n}", flush=True)
        return torch.cat(out, dim=0) if out else torch.zeros((0, self.dim))
