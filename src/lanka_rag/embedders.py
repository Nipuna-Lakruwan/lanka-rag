"""Dense embedders (Ollama and Hugging Face) with an on-disk cache."""

from __future__ import annotations

import hashlib
import os
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

import httpx
import numpy as np

from .models import EmbeddingModel

DEFAULT_OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
if not DEFAULT_OLLAMA_HOST.startswith("http"):
    DEFAULT_OLLAMA_HOST = f"http://{DEFAULT_OLLAMA_HOST}"


class Embedder(Protocol):
    model: EmbeddingModel

    def embed(self, texts: Sequence[str]) -> np.ndarray: ...


def l2_normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(norms, 1e-12, None)


class OllamaEmbedder:
    def __init__(
        self,
        model: EmbeddingModel,
        host: str = DEFAULT_OLLAMA_HOST,
        batch_size: int = 16,
        timeout: float = 600.0,
    ) -> None:
        self.model = model
        self.host = host.rstrip("/")
        self.batch_size = batch_size
        self.client = httpx.Client(timeout=timeout)

    def available(self) -> bool:
        try:
            tags = self.client.get(f"{self.host}/api/tags").json().get("models", [])
        except httpx.HTTPError:
            return False
        want = (
            self.model.model_id if ":" in self.model.model_id else f"{self.model.model_id}:latest"
        )
        return any(m.get("name") == want for m in tags)

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        vectors: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = list(texts[i : i + self.batch_size])
            r = self.client.post(
                f"{self.host}/api/embed",
                json={"model": self.model.model_id, "input": batch, "truncate": True},
            )
            r.raise_for_status()
            vectors.extend(r.json()["embeddings"])
        return l2_normalize(np.asarray(vectors, dtype=np.float32))


class HFEmbedder:
    def __init__(self, model: EmbeddingModel, batch_size: int = 16) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:  # pragma: no cover - optional dependency
            raise SystemExit("Install the hf extra: uv sync --extra hf") from e
        self.model = model
        self.batch_size = batch_size
        self.st = SentenceTransformer(model.model_id)

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        vecs = self.st.encode(
            list(texts),
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(vecs, dtype=np.float32)


def make_embedder(model: EmbeddingModel, **kwargs) -> Embedder:
    if model.backend == "ollama":
        return OllamaEmbedder(model, **kwargs)
    if model.backend == "hf":
        return HFEmbedder(model, **kwargs)
    raise ValueError(f"{model.key} is not a dense model (backend={model.backend})")


class CachedEmbedder:
    """Caches whole batches on disk, keyed by model id and exact input texts.

    Benchmark reruns (new metrics, new report) then cost nothing, and you can
    ``ollama rm`` a model once it has been scored.
    """

    def __init__(self, inner: Embedder, cache_dir: Path) -> None:
        self.inner = inner
        self.model = inner.model
        self.dir = cache_dir / self.model.key
        self.dir.mkdir(parents=True, exist_ok=True)
        self.last_seconds = 0.0
        self.last_was_cached = False

    def _path(self, texts: Sequence[str]) -> Path:
        h = hashlib.sha256(self.model.model_id.encode())
        for t in texts:
            h.update(b"\x00")
            h.update(t.encode())
        return self.dir / f"{h.hexdigest()[:24]}.npy"

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """Embed texts; ``last_seconds`` is the original (uncached) wall time."""
        path = self._path(texts)
        timing = path.with_suffix(".seconds")
        if path.exists():
            self.last_was_cached = True
            self.last_seconds = float(timing.read_text()) if timing.exists() else 0.0
            return np.load(path)
        start = time.perf_counter()
        vecs = self.inner.embed(texts)
        self.last_seconds = time.perf_counter() - start
        self.last_was_cached = False
        np.save(path, vecs)
        timing.write_text(f"{self.last_seconds:.3f}")
        return vecs
