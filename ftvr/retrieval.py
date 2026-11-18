"""Dense (MiniLM) and sparse (BM25) retrieval over the labelled training pool."""

from __future__ import annotations

from collections import Counter

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from .data import Example


class Embedder:
    """sentence-transformers/all-MiniLM-L6-v2 with mean pooling + L2 norm (no s-t dependency)."""

    def __init__(self, model_dir: str = "models/all-MiniLM-L6-v2"):
        self.tok = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModel.from_pretrained(model_dir).eval()

    @torch.inference_mode()
    def encode(self, texts: list[str], batch_size: int = 128) -> np.ndarray:
        out = []
        for i in range(0, len(texts), batch_size):
            b = self.tok(texts[i : i + batch_size], padding=True, truncation=True, max_length=128, return_tensors="pt")
            h = self.model(**b).last_hidden_state
            m = b["attention_mask"].unsqueeze(-1).float()
            v = (h * m).sum(1) / m.sum(1).clamp(min=1e-9)
            out.append(torch.nn.functional.normalize(v, dim=-1).numpy())
        return np.concatenate(out) if out else np.zeros((0, 384), dtype=np.float32)


class DenseIndex:
    def __init__(self, pool: list[Example], embedder: Embedder, pool_vecs: np.ndarray | None = None):
        self.pool = pool
        self.emb = embedder
        self.vecs = pool_vecs if pool_vecs is not None else embedder.encode([e.text for e in pool])

    def search(self, queries: list[str], k: int, query_vecs: np.ndarray | None = None, exclude_self: bool = False):
        q = query_vecs if query_vecs is not None else self.emb.encode(queries)
        sims = q @ self.vecs.T
        if exclude_self:  # used when building RAFT-style training prompts from the pool itself
            np.fill_diagonal(sims, -np.inf)
        top = np.argsort(-sims, axis=1)[:, :k]
        return [[self.pool[j] for j in row] for row in top], np.take_along_axis(sims, top, 1)


def candidates_from_shots(shots: list[Example], max_labels: int | None = None) -> list[str]:
    """Distinct labels among retrieved shots, most frequent (then earliest) first."""
    cnt = Counter(s.label for s in shots)
    first = {}
    for i, s in enumerate(shots):
        first.setdefault(s.label, i)
    labs = sorted(cnt, key=lambda lab: (-cnt[lab], first[lab]))
    return labs[:max_labels] if max_labels else labs


def knn_vote(shots: list[Example], sims: np.ndarray) -> str:
    score: dict[str, float] = {}
    for s, w in zip(shots, sims):
        score[s.label] = score.get(s.label, 0.0) + float(w)
    return max(score, key=score.get)
