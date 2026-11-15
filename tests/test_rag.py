from pathlib import Path

from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from lanka_rag.rag.chain import answer_language, ask, build_chain, format_docs
from lanka_rag.rag.ingest import load_documents
from lanka_rag.rag.store import LankaIndex, reciprocal_rank_fusion

DOCS = {
    "colombo.txt": "කොළඹ ශ්‍රී ලංකාවේ විශාලතම නගරය සහ ප්‍රධාන වරාය වේ.\n\nකොළඹ වරාය 1912 දී පුළුල් කරන ලදී.",
    "kandy.md": "# මහනුවර\n\nමහනුවර දළදා මාළිගාව පිහිටා ඇත. එය 1592 දී ගොඩනගන ලදී.",
    "jaffna.html": "<html><body><script>x=1</script><p>யாழ்ப்பாணம் பொது நூலகம் 1933 இல் "
    "நிறுவப்பட்டது.</p></body></html>",
    "legacy.txt": "Y%S ,xld m%cd;dka;%sl iudcjd§ ckrcfha .eiÜ m;%h wxl 2345$12 - 2023 cqks ui 05 "
    "jeks i÷od - 2023 rcfha ksfjokh",
}


def _write_corpus(tmp_path: Path) -> Path:
    d = tmp_path / "docs"
    d.mkdir()
    for name, text in DOCS.items():
        (d / name).write_text(text, encoding="utf-8")
    return d


def test_ingest_detects_language_and_skips_legacy(tmp_path: Path):
    docs, rep = load_documents([_write_corpus(tmp_path)], chunk_size=200, chunk_overlap=0)
    assert rep.files == 4
    assert rep.skipped_legacy == ["legacy.txt"]
    assert rep.languages == {"si": 2, "ta": 1}
    assert all("x=1" not in d.page_content for d in docs)


def test_rrf():
    fused = reciprocal_rank_fusion([[1, 2, 3], [3, 1]])
    assert [i for i, _ in fused][:2] == [1, 3]


def test_index_roundtrip_and_chain(tmp_path: Path):
    docs, _ = load_documents([_write_corpus(tmp_path)], chunk_size=200, chunk_overlap=0)
    emb = DeterministicFakeEmbedding(size=32)
    index = LankaIndex.build(docs, "bge-m3", embeddings=emb)
    index.save(tmp_path / "idx")
    loaded = LankaIndex.load(tmp_path / "idx", embeddings=emb)
    assert len(loaded.docs) == len(docs)

    # BM25 alone must find the Kandy chunk; the fake dense ranking is random.
    hits = loaded.retriever(k=2, mode="bm25").invoke("දළදා මාළිගාව කොහෙද?")
    assert "kandy.md" in hits[0].metadata["source"]

    # "1912" occurs only in a Sinhala chunk. Pure BM25 matches it; hybrid mode must not
    # let an English question pull that chunk in through lexical noise.
    sinhala_1912 = next(i for i, d in enumerate(loaded.docs) if "1912" in d.page_content)
    assert (
        loaded.retriever(mode="bm25").lexical_ranking("What happened in 1912?")[0] == sinhala_1912
    )
    assert loaded.retriever(mode="hybrid").lexical_ranking("What happened in 1912?") == []
    # Same-language lexical matching still works in hybrid mode.
    assert loaded.retriever(mode="hybrid").lexical_ranking("1912 කොළඹ වරාය")[0] == sinhala_1912

    llm = FakeListChatModel(responses=["මහනුවර [1]"])
    chain = build_chain(loaded.retriever(k=3), llm)
    ans = ask(chain, "දළදා මාළිගාව කොහෙද?")
    assert ans.answer == "මහනුවර [1]"
    assert 1 <= len(ans.sources) <= 3


def test_prompt_helpers():
    assert answer_language("யாழ்ப்பாணம் நூலகம் எப்போது?") == "Tamil"
    assert answer_language("When was it built?") == "English"
    from langchain_core.documents import Document

    text = format_docs([Document(page_content="abc", metadata={"source": "/x/a.pdf", "page": 3})])
    assert text == "[1] (a.pdf, p.3)\nabc"
