"""Retrieval tasks built from Belebele (CC BY-SA 4.0).

Belebele has 900 human-written questions over 488 FLORES-200 passages, and the
same questions and passages exist in Sinhala script, romanized Sinhala
("Singlish"), Tamil and English. Because every language is aligned by passage,
one dataset gives monolingual, cross-lingual and transliterated retrieval tasks
with identical relevance labels.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import httpx

from ..text import ZWJ, normalize

BELEBELE_URL = "https://huggingface.co/datasets/facebook/belebele/resolve/main/data/{config}.jsonl"

# Our language codes -> Belebele configs.
LANGS = {
    "si": "sin_Sinh",  # Sinhala, Sinhala script
    "si-Latn": "sin_Latn",  # Sinhala, romanized ("Singlish")
    "ta": "tam_Taml",  # Tamil
    "en": "eng_Latn",  # English
}

LANG_LABELS = {"si": "Sinhala", "si-Latn": "Singlish", "ta": "Tamil", "en": "English"}


@dataclass
class RetrievalTask:
    name: str
    description: str
    query_lang: str
    doc_lang: str
    query_ids: list[str]
    queries: list[str]
    doc_ids: list[str]
    docs: list[str]
    qrels: dict[str, set[str]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        assert len(self.query_ids) == len(self.queries)
        assert len(self.doc_ids) == len(self.docs)


@dataclass(frozen=True)
class TaskSpec:
    name: str
    query_lang: str
    doc_lang: str
    description: str
    query_transform: str | None = None  # e.g. "strip_zwj"


TASKS: list[TaskSpec] = [
    TaskSpec("si→si", "si", "si", "Sinhala question, Sinhala passages"),
    TaskSpec(
        "si→si (no ZWJ)",
        "si",
        "si",
        "Sinhala question typed without zero width joiners (common on phone keyboards)",
        query_transform="strip_zwj",
    ),
    TaskSpec("si-Latn→si", "si-Latn", "si", "Singlish (romanized) question, Sinhala passages"),
    TaskSpec("ta→ta", "ta", "ta", "Tamil question, Tamil passages"),
    TaskSpec("en→en", "en", "en", "English reference: how good is the model at all?"),
    TaskSpec("en→si", "en", "si", "English question, Sinhala passages"),
    TaskSpec("si→en", "si", "en", "Sinhala question, English passages"),
    TaskSpec("ta→si", "ta", "si", "Tamil question, Sinhala passages"),
    TaskSpec("si→ta", "si", "ta", "Sinhala question, Tamil passages"),
]


def download_belebele(data_dir: Path) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    with httpx.Client(follow_redirects=True, timeout=60) as client:
        for config in LANGS.values():
            path = data_dir / f"belebele_{config}.jsonl"
            if path.exists():
                continue
            r = client.get(BELEBELE_URL.format(config=config))
            r.raise_for_status()
            path.write_bytes(r.content)


def _load_rows(data_dir: Path, lang: str) -> list[dict]:
    path = data_dir / f"belebele_{LANGS[lang]}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing. Run: lanka-rag bench download")
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _passages(rows: list[dict]) -> dict[str, str]:
    # The source link identifies a passage across all languages.
    out: dict[str, str] = {}
    for r in rows:
        out.setdefault(r["link"], normalize(r["flores_passage"]))
    return out


def _transform(text: str, how: str | None) -> str:
    if how is None:
        return text
    if how == "strip_zwj":
        return text.replace(ZWJ, "")
    raise ValueError(how)


def build_task(spec: TaskSpec, data_dir: Path, limit: int | None = None) -> RetrievalTask:
    q_rows = _load_rows(data_dir, spec.query_lang)
    passages = _passages(_load_rows(data_dir, spec.doc_lang))
    doc_ids = sorted(passages)
    if limit:
        q_rows = q_rows[:limit]
    qids = [f"{r['link']}#{r['question_number']}" for r in q_rows]
    queries = [_transform(normalize(r["question"]), spec.query_transform) for r in q_rows]
    qrels = {qid: {r["link"]} for qid, r in zip(qids, q_rows, strict=True)}
    return RetrievalTask(
        name=spec.name,
        description=spec.description,
        query_lang=spec.query_lang,
        doc_lang=spec.doc_lang,
        query_ids=qids,
        queries=queries,
        doc_ids=doc_ids,
        docs=[passages[d] for d in doc_ids],
        qrels=qrels,
    )


def build_tasks(
    data_dir: Path, names: list[str] | None = None, limit: int | None = None
) -> list[RetrievalTask]:
    specs = [s for s in TASKS if names is None or s.name in names]
    if names:
        unknown = set(names) - {s.name for s in specs}
        if unknown:
            raise SystemExit(
                f"Unknown task(s): {sorted(unknown)}. Known: {[s.name for s in TASKS]}"
            )
    return [build_task(s, data_dir, limit) for s in specs]
