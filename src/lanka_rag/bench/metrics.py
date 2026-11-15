"""Ranking metrics. Every Belebele query has exactly one relevant passage, but the
functions accept sets so they also work for multi-label tasks."""

from __future__ import annotations

import math

import numpy as np


def rank_of_first_relevant(ranked: list[str], relevant: set[str]) -> int | None:
    """1-based rank of the first relevant doc, or None if absent."""
    for i, d in enumerate(ranked, start=1):
        if d in relevant:
            return i
    return None


def recall_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    return len(set(ranked[:k]) & relevant) / len(relevant)


def mrr_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    r = rank_of_first_relevant(ranked[:k], relevant)
    return 1.0 / r if r else 0.0


def ndcg_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    dcg = sum(1.0 / math.log2(i + 1) for i, d in enumerate(ranked[:k], start=1) if d in relevant)
    ideal = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(relevant), k) + 1))
    return dcg / ideal if ideal else 0.0


def top_k_ids(scores: np.ndarray, doc_ids: list[str], k: int) -> list[list[str]]:
    """Top-k doc ids per row of a (queries x docs) score matrix, best first.

    Ties break by corpus order (stable sort), so results are reproducible across
    machines; BM25 gives many exact ties (often zero) on cross-lingual tasks.
    """
    k = min(k, scores.shape[1])
    idx = np.argsort(-scores, axis=1, kind="stable")[:, :k]
    return [[doc_ids[j] for j in row] for row in idx]


def evaluate(
    ranked: dict[str, list[str]], qrels: dict[str, set[str]], ks: tuple[int, ...] = (1, 5, 10)
) -> dict[str, float]:
    """Mean metrics over queries. ``ranked`` maps query id -> ranked doc ids."""
    out: dict[str, list[float]] = {"ndcg@10": [], "mrr@10": []}
    for k in ks:
        out[f"recall@{k}"] = []
    for qid, rel in qrels.items():
        r = ranked.get(qid, [])
        out["ndcg@10"].append(ndcg_at_k(r, rel, 10))
        out["mrr@10"].append(mrr_at_k(r, rel, 10))
        for k in ks:
            out[f"recall@{k}"].append(recall_at_k(r, rel, k))
    return {m: float(np.mean(v)) if v else 0.0 for m, v in out.items()}


def bootstrap_ci(
    per_query: list[float], n_boot: int = 1000, alpha: float = 0.05, seed: int = 0
) -> tuple[float, float]:
    """Percentile bootstrap confidence interval for a mean."""
    if not per_query:
        return (0.0, 0.0)
    rng = np.random.default_rng(seed)
    arr = np.asarray(per_query)
    means = rng.choice(arr, size=(n_boot, len(arr)), replace=True).mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)
