"""Registry of embedding models and the prompt format each one was trained with.

Getting the query/document prefixes wrong is the most common reason a
multilingual embedder looks worse than it is, so they live here in one place
and are shared by the benchmark and the RAG app.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EmbeddingModel:
    key: str  # short name used on the CLI and in result files
    backend: str  # "ollama" | "hf" | "bm25"
    model_id: str  # Ollama tag, Hugging Face repo id, or BM25 variant
    query_prefix: str = ""
    doc_prefix: str = ""
    params: str = ""  # rough parameter count, for the report
    multilingual: bool = True
    notes: str = ""

    def format_query(self, text: str) -> str:
        return f"{self.query_prefix}{text}"

    def format_doc(self, text: str) -> str:
        return f"{self.doc_prefix}{text}"


# Format from the Qwen3-Embedding model card: f"Instruct: {task}\nQuery:{query}"
_QWEN3_INSTRUCT = "Instruct: Given a question, retrieve the passage that answers it\nQuery:"

MODELS: dict[str, EmbeddingModel] = {
    m.key: m
    for m in [
        # Lexical baselines (no model download).
        EmbeddingModel("bm25-word", "bm25", "word", params="-", notes="Unicode-aware tokens"),
        EmbeddingModel("bm25-char", "bm25", "char", params="-", notes="char 3-4 grams"),
        # Ollama library models.
        EmbeddingModel("bge-m3", "ollama", "bge-m3", params="568M"),
        EmbeddingModel(
            "qwen3-embedding-0.6b",
            "ollama",
            "qwen3-embedding:0.6b",
            query_prefix=_QWEN3_INSTRUCT,
            params="0.6B",
        ),
        EmbeddingModel(
            "qwen3-embedding-4b",
            "ollama",
            "qwen3-embedding:4b",
            query_prefix=_QWEN3_INSTRUCT,
            params="4B",
        ),
        EmbeddingModel(
            "embeddinggemma",
            "ollama",
            "embeddinggemma",
            query_prefix="task: search result | query: ",
            doc_prefix="title: none | text: ",
            params="308M",
        ),
        EmbeddingModel(
            "snowflake-arctic-embed2",
            "ollama",
            "snowflake-arctic-embed2",
            query_prefix="query: ",
            params="568M",
        ),
        EmbeddingModel("granite-embedding", "ollama", "granite-embedding:278m", params="278M"),
        EmbeddingModel(
            "paraphrase-multilingual", "ollama", "paraphrase-multilingual", params="278M"
        ),
        EmbeddingModel(
            "nomic-embed-text",
            "ollama",
            "nomic-embed-text",
            query_prefix="search_query: ",
            doc_prefix="search_document: ",
            params="137M",
            multilingual=False,
            notes="English-centric control",
        ),
        # Hugging Face models (pip install 'lanka-rag[hf]').
        EmbeddingModel(
            "multilingual-e5-base",
            "hf",
            "intfloat/multilingual-e5-base",
            query_prefix="query: ",
            doc_prefix="passage: ",
            params="278M",
        ),
        EmbeddingModel(
            "multilingual-e5-large",
            "hf",
            "intfloat/multilingual-e5-large",
            query_prefix="query: ",
            doc_prefix="passage: ",
            params="560M",
        ),
        EmbeddingModel("labse", "hf", "sentence-transformers/LaBSE", params="471M"),
    ]
}


def get_model(key: str) -> EmbeddingModel:
    try:
        return MODELS[key]
    except KeyError:
        known = ", ".join(MODELS)
        raise SystemExit(f"Unknown model '{key}'. Known models: {known}") from None
