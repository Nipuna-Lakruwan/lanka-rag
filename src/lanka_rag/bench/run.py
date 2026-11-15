"""Run retrieval tasks for one or more models and write per-model JSON results."""

from __future__ import annotations

import json
import platform
import time
from datetime import UTC, datetime
from pathlib import Path

import httpx
import numpy as np
from rich.console import Console

from ..bm25 import BM25
from ..embedders import DEFAULT_OLLAMA_HOST, CachedEmbedder, OllamaEmbedder, make_embedder
from ..models import EmbeddingModel, get_model
from .datasets import RetrievalTask, build_tasks
from .metrics import bootstrap_ci, evaluate, ndcg_at_k, rank_of_first_relevant, top_k_ids

console = Console()
TOP_K = 100


def ollama_pull(model_id: str, host: str = DEFAULT_OLLAMA_HOST) -> None:
    console.print(f"[cyan]Pulling {model_id} from the Ollama library...[/]")
    with httpx.Client(timeout=None) as c:
        r = c.post(f"{host}/api/pull", json={"model": model_id, "stream": False})
        r.raise_for_status()


def ollama_rm(model_id: str, host: str = DEFAULT_OLLAMA_HOST) -> None:
    with httpx.Client(timeout=60) as c:
        c.request("DELETE", f"{host}/api/delete", json={"model": model_id}).raise_for_status()
    console.print(f"[dim]Removed {model_id} from Ollama.[/]")


def _score_bm25(model: EmbeddingModel, task: RetrievalTask) -> tuple[np.ndarray, dict]:
    start = time.perf_counter()
    index = BM25(task.docs, analyzer=model.model_id)
    scores = index.score_matrix(task.queries)
    return scores, {"seconds": round(time.perf_counter() - start, 3)}


def _score_dense(emb: CachedEmbedder, task: RetrievalTask) -> tuple[np.ndarray, dict]:
    m = emb.model
    d = emb.embed([m.format_doc(t) for t in task.docs])
    doc_s = emb.last_seconds
    q = emb.embed([m.format_query(t) for t in task.queries])
    q_s = emb.last_seconds
    timing = {
        "doc_seconds": round(doc_s, 3),
        "query_seconds": round(q_s, 3),
        "docs_per_second": round(len(task.docs) / doc_s, 2) if doc_s else None,
        "dim": int(d.shape[1]),
    }
    return q @ d.T, timing


def run_model(
    key: str,
    tasks: list[RetrievalTask],
    out_dir: Path,
    cache_dir: Path,
    *,
    pull: bool = False,
    rm_after: bool = False,
    host: str = DEFAULT_OLLAMA_HOST,
) -> dict:
    model = get_model(key)
    emb: CachedEmbedder | None = None
    pulled_here = False
    if model.backend != "bm25":
        inner = (
            make_embedder(model, host=host) if model.backend == "ollama" else make_embedder(model)
        )
        if isinstance(inner, OllamaEmbedder) and not inner.available():
            if not pull:
                raise SystemExit(
                    f"{model.model_id} is not in Ollama. Pass --pull or `ollama pull`."
                )
            ollama_pull(model.model_id, host)
            pulled_here = True
        emb = CachedEmbedder(inner, cache_dir)

    results = []
    try:
        for task in tasks:
            t0 = time.perf_counter()
            scores, timing = _score_bm25(model, task) if emb is None else _score_dense(emb, task)
            ranked_lists = top_k_ids(scores, task.doc_ids, TOP_K)
            ranked = dict(zip(task.query_ids, ranked_lists, strict=True))
            metrics = evaluate(ranked, task.qrels)
            per_q = [ndcg_at_k(ranked[q], task.qrels[q], 10) for q in task.query_ids]
            lo, hi = bootstrap_ci(per_q)
            ranks = {q: rank_of_first_relevant(ranked[q], task.qrels[q]) for q in task.query_ids}
            results.append(
                {
                    "task": task.name,
                    "query_lang": task.query_lang,
                    "doc_lang": task.doc_lang,
                    "n_queries": len(task.queries),
                    "n_docs": len(task.docs),
                    "metrics": metrics,
                    "ndcg@10_ci95": [lo, hi],
                    "timing": timing,
                    "ranks": ranks,
                }
            )
            console.print(
                f"  {model.key:<26} {task.name:<16} nDCG@10={metrics['ndcg@10']:.3f} "
                f"R@1={metrics['recall@1']:.3f}  ({time.perf_counter() - t0:.1f}s)"
            )
    finally:
        if pulled_here and rm_after:
            ollama_rm(model.model_id, host)

    record = {
        "model": {
            "key": model.key,
            "backend": model.backend,
            "model_id": model.model_id,
            "params": model.params,
            "multilingual": model.multilingual,
            "query_prefix": model.query_prefix,
            "doc_prefix": model.doc_prefix,
            "notes": model.notes,
        },
        "results": results,
        "env": {
            "date": datetime.now(UTC).isoformat(timespec="seconds"),
            "machine": platform.machine(),
            "platform": platform.platform(terse=True),
        },
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{model.key}.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    return record


def run(
    model_keys: list[str],
    data_dir: Path,
    out_dir: Path,
    cache_dir: Path,
    task_names: list[str] | None = None,
    limit: int | None = None,
    **kwargs,
) -> None:
    tasks = build_tasks(data_dir, task_names, limit)
    console.print(
        f"[bold]{len(tasks)} tasks[/], {len(tasks[0].queries)} queries x "
        f"{len(tasks[0].docs)} passages each"
    )
    for key in model_keys:
        console.rule(key)
        try:
            run_model(key, tasks, out_dir, cache_dir, **kwargs)
        except (httpx.HTTPError, SystemExit) as e:
            console.print(f"[red]Skipping {key}: {e}[/]")
