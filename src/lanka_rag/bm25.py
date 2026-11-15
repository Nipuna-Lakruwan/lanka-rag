"""Okapi BM25 over Sinhala/Tamil-aware tokens or character n-grams."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Callable, Sequence

import numpy as np

from .text import char_ngrams, tokenize

Analyzer = Callable[[str], list[str]]

ANALYZERS: dict[str, Analyzer] = {"word": tokenize, "char": char_ngrams}


class BM25:
    def __init__(
        self,
        docs: Sequence[str],
        analyzer: str | Analyzer = "word",
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self.analyze: Analyzer = ANALYZERS[analyzer] if isinstance(analyzer, str) else analyzer
        self.k1, self.b = k1, b
        self.doc_tfs: list[Counter[str]] = [Counter(self.analyze(d)) for d in docs]
        self.doc_lens = np.array([sum(tf.values()) for tf in self.doc_tfs], dtype=np.float64)
        self.avgdl = float(self.doc_lens.mean()) if len(docs) else 0.0
        df: Counter[str] = Counter()
        for tf in self.doc_tfs:
            df.update(tf.keys())
        n = len(docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}
        # Inverted index: term -> [(doc_id, tf), ...]
        self.postings: dict[str, list[tuple[int, int]]] = {}
        for i, tf in enumerate(self.doc_tfs):
            for t, c in tf.items():
                self.postings.setdefault(t, []).append((i, c))

    def __len__(self) -> int:
        return len(self.doc_tfs)

    def scores(self, query: str) -> np.ndarray:
        out = np.zeros(len(self.doc_tfs), dtype=np.float64)
        if not self.avgdl:
            return out
        for term, qtf in Counter(self.analyze(query)).items():
            idf = self.idf.get(term)
            if idf is None:
                continue
            for i, tf in self.postings[term]:
                norm = self.k1 * (1 - self.b + self.b * self.doc_lens[i] / self.avgdl)
                out[i] += qtf * idf * tf * (self.k1 + 1) / (tf + norm)
        return out

    def score_matrix(self, queries: Sequence[str]) -> np.ndarray:
        return np.vstack([self.scores(q) for q in queries]) if queries else np.zeros((0, len(self)))
