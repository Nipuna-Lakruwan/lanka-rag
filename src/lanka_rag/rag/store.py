"""Hybrid (dense + BM25) index built on LangChain primitives."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import OllamaEmbeddings
from pydantic import ConfigDict

from ..bm25 import BM25
from ..embedders import DEFAULT_OLLAMA_HOST
from ..models import EmbeddingModel, get_model
from ..text import detect_language

INDEX_VERSION = 1


class PrefixedOllamaEmbeddings(Embeddings):
    """OllamaEmbeddings that applies the query/document prompt the model expects."""

    def __init__(self, model: EmbeddingModel, base_url: str = DEFAULT_OLLAMA_HOST) -> None:
        if model.backend != "ollama":
            raise ValueError(f"{model.key} is not an Ollama model")
        self.model = model
        self.inner = OllamaEmbeddings(model=model.model_id, base_url=base_url)

    def embed_documents(self, texts: list[str], batch_size: int = 32) -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(texts), batch_size):
            batch = [self.model.format_doc(t) for t in texts[i : i + batch_size]]
            out.extend(self.inner.embed_documents(batch))
        return out

    def embed_query(self, text: str) -> list[float]:
        return self.inner.embed_query(self.model.format_query(text))


def reciprocal_rank_fusion(rankings: list[list[int]], k: int = 60) -> list[tuple[int, float]]:
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, doc_idx in enumerate(ranking, start=1):
            scores[doc_idx] = scores.get(doc_idx, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)


class HybridRetriever(BaseRetriever):
    """Dense retrieval and character n-gram BM25, fused with RRF.

    The benchmark shows why both: char BM25 is strong within one language,
    dense models are the only thing that works across languages.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    vectorstore: InMemoryVectorStore
    bm25: BM25
    docs: list[Document]
    k: int = 5
    candidates: int = 30
    mode: str = "hybrid"  # "hybrid" | "dense" | "bm25"

    def lexical_ranking(self, query: str) -> list[int]:
        """BM25 ranking of chunk indices.

        In hybrid mode only chunks in the question's language are scored. BM25 does
        not work across languages (nDCG@10 ~0.05 cross-lingual in the benchmark), so
        an English question over Sinhala chunks would otherwise fuse in noise that
        matched on numbers or stray Latin letters.
        """
        scores = self.bm25.scores(query)
        if self.mode == "hybrid":
            q_lang = detect_language(query)
            same = np.array([d.metadata.get("lang") == q_lang for d in self.docs])
            scores = np.where(same, scores, 0.0)
        order = np.argsort(-scores, kind="stable")[: self.candidates]
        return [int(i) for i in order if scores[i] > 0]

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        rankings: list[list[int]] = []
        if self.mode in {"hybrid", "dense"}:
            hits = self.vectorstore.similarity_search(query, k=self.candidates)
            rankings.append([int(d.id) for d in hits if d.id is not None])
        if self.mode in {"hybrid", "bm25"}:
            rankings.append(self.lexical_ranking(query))
        out = []
        for idx, score in reciprocal_rank_fusion(rankings)[: self.k]:
            d = self.docs[idx]
            out.append(Document(page_content=d.page_content, metadata={**d.metadata, "rrf": score}))
        return out


class LankaIndex:
    def __init__(self, embedding_key: str, docs: list[Document], vectorstore: InMemoryVectorStore):
        self.embedding_key = embedding_key
        self.docs = docs
        self.vectorstore = vectorstore
        self.bm25 = BM25([d.page_content for d in docs], analyzer="char")

    @classmethod
    def build(
        cls,
        docs: list[Document],
        embedding_key: str,
        base_url: str = DEFAULT_OLLAMA_HOST,
        embeddings: Embeddings | None = None,  # override, e.g. a fake in tests
    ) -> LankaIndex:
        emb = embeddings or PrefixedOllamaEmbeddings(get_model(embedding_key), base_url)
        vs = InMemoryVectorStore(embedding=emb)
        vs.add_documents(docs, ids=[str(i) for i in range(len(docs))])
        return cls(embedding_key, docs, vs)

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        self.vectorstore.dump(str(directory / "vectors.json"))
        meta = {"version": INDEX_VERSION, "embedding": self.embedding_key, "n_docs": len(self.docs)}
        (directory / "index.json").write_text(json.dumps(meta, indent=1))

    @classmethod
    def load(
        cls,
        directory: Path,
        base_url: str = DEFAULT_OLLAMA_HOST,
        embeddings: Embeddings | None = None,
    ) -> LankaIndex:
        meta_path = directory / "index.json"
        if not meta_path.exists():
            raise SystemExit(f"No index at {directory}. Run `lanka-rag ingest` first.")
        meta = json.loads(meta_path.read_text())
        emb = embeddings or PrefixedOllamaEmbeddings(get_model(meta["embedding"]), base_url)
        vs = InMemoryVectorStore.load(str(directory / "vectors.json"), embedding=emb)
        ids = sorted(vs.store, key=int)
        docs = [
            Document(page_content=vs.store[i]["text"], metadata=vs.store[i]["metadata"], id=i)
            for i in ids
        ]
        return cls(meta["embedding"], docs, vs)

    def retriever(self, k: int = 5, mode: str = "hybrid") -> HybridRetriever:
        return HybridRetriever(
            vectorstore=self.vectorstore, bm25=self.bm25, docs=self.docs, k=k, mode=mode
        )
