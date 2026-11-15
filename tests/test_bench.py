import json
from pathlib import Path

import numpy as np
import pytest

from lanka_rag.bench import datasets
from lanka_rag.bench.metrics import (
    bootstrap_ci,
    evaluate,
    mrr_at_k,
    ndcg_at_k,
    recall_at_k,
    top_k_ids,
)
from lanka_rag.bench.report import heatmap_svg, markdown_tables
from lanka_rag.bench.run import run_model
from lanka_rag.bm25 import BM25


def test_metrics_single_relevant():
    ranked = ["a", "b", "c"]
    assert recall_at_k(ranked, {"b"}, 1) == 0
    assert recall_at_k(ranked, {"b"}, 2) == 1
    assert mrr_at_k(ranked, {"c"}, 10) == pytest.approx(1 / 3)
    assert ndcg_at_k(ranked, {"a"}, 10) == 1
    assert ndcg_at_k(ranked, {"b"}, 10) == pytest.approx(1 / np.log2(3))
    assert ndcg_at_k(ranked, {"z"}, 10) == 0


def test_evaluate_and_ci():
    m = evaluate({"q1": ["a", "b"], "q2": ["b", "a"]}, {"q1": {"a"}, "q2": {"a"}})
    assert m["recall@1"] == 0.5
    assert m["recall@5"] == 1.0
    lo, hi = bootstrap_ci([0.0, 1.0] * 50)
    assert 0.35 < lo < 0.5 < hi < 0.65


def test_top_k_ids_orders_best_first():
    scores = np.array([[0.1, 0.9, 0.5], [0.3, 0.2, 0.1]])
    assert top_k_ids(scores, ["a", "b", "c"], 2) == [["b", "c"], ["a", "b"]]


def test_bm25_prefers_matching_doc():
    docs = ["කොළඹ නගරය", "මහනුවර දළදා මාළිගාව", "யாழ்ப்பாணம் நூலகம்"]
    for analyzer in ("word", "char"):
        bm = BM25(docs, analyzer=analyzer)
        assert int(np.argmax(bm.scores("දළදා මාළිගාව කොහෙද?"))) == 1
        assert int(np.argmax(bm.scores("யாழ்ப்பாணம்"))) == 2


@pytest.fixture
def tiny_belebele(tmp_path: Path) -> Path:
    passages = {
        "l1": {"si": "කොළඹ ශ්‍රී ලංකාවේ විශාලතම නගරයයි.", "en": "Colombo is the largest city.",
               "ta": "கொழும்பு மிகப்பெரிய நகரம்.", "si-Latn": "Colombo lankawe wishalathama nagaraya."},
        "l2": {"si": "මහනුවර දළදා මාළිගාව පිහිටා ඇත.", "en": "Kandy has the Temple of the Tooth.",
               "ta": "கண்டியில் தலதா மாளிகை உள்ளது.", "si-Latn": "Mahanuwara dalada maligawa thiyenawa."},
    }  # fmt: skip
    questions = {
        "l1": {"si": "විශාලතම නගරය කුමක්ද?", "en": "Which is the largest city?",
               "ta": "மிகப்பெரிய நகரம் எது?", "si-Latn": "wishalathama nagaraya mokakda?"},
        "l2": {"si": "දළදා මාළිගාව කොහෙද?", "en": "Where is the Temple of the Tooth?",
               "ta": "தலதா மாளிகை எங்கே?", "si-Latn": "dalada maligawa koheda?"},
    }  # fmt: skip
    for lang, config in datasets.LANGS.items():
        with (tmp_path / f"belebele_{config}.jsonl").open("w", encoding="utf-8") as f:
            for link in passages:
                row = {"link": link, "question_number": 1,
                       "flores_passage": passages[link][lang], "question": questions[link][lang]}  # fmt: skip
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return tmp_path


def test_build_tasks_aligns_languages(tiny_belebele: Path):
    tasks = {t.name: t for t in datasets.build_tasks(tiny_belebele)}
    assert set(tasks) == {s.name for s in datasets.TASKS}
    t = tasks["en→si"]
    assert t.doc_ids == ["l1", "l2"]
    assert t.qrels == {"l1#1": {"l1"}, "l2#1": {"l2"}}
    assert "‍" not in "".join(tasks["si→si (no ZWJ)"].queries)


def test_run_model_bm25_and_report(tiny_belebele: Path, tmp_path: Path):
    tasks = datasets.build_tasks(tiny_belebele, ["si→si", "ta→ta"])
    rec = run_model("bm25-char", tasks, tmp_path / "raw", tmp_path / "cache")
    assert {r["task"] for r in rec["results"]} == {"si→si", "ta→ta"}
    assert all(r["metrics"]["recall@1"] == 1.0 for r in rec["results"])
    assert (tmp_path / "raw" / "bm25-char.json").exists()
    md = markdown_tables([rec])
    assert "bm25-char" in md and "si→si" in md
    svg = heatmap_svg([rec], "dark")
    assert svg.startswith("<svg") and "bm25-char" in svg
