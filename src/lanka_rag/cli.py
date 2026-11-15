"""Command line interface: `lanka-rag bench ...`, `ingest`, `ask`, `chat`, `inspect`."""

from __future__ import annotations

import argparse
from pathlib import Path

from rich.console import Console
from rich.table import Table

from .embedders import DEFAULT_OLLAMA_HOST
from .models import MODELS

console = Console()

DATA_DIR = Path("data/raw")
RESULTS_DIR = Path("results")
CACHE_DIR = Path(".cache/embeddings")
INDEX_DIR = Path(".index")
DEFAULT_EMBEDDING = "bge-m3"


def _bench(args: argparse.Namespace) -> None:
    from .bench import datasets, report, run

    if args.bench_cmd == "download":
        datasets.download_belebele(args.data)
        console.print(f"Belebele (si, si-Latn, ta, en) saved to {args.data}")
    elif args.bench_cmd == "models":
        t = Table("key", "backend", "model id", "params", "query prefix", "doc prefix")
        for m in MODELS.values():
            t.add_row(
                m.key, m.backend, m.model_id, m.params, repr(m.query_prefix), repr(m.doc_prefix)
            )
        console.print(t)
    elif args.bench_cmd == "run":
        run.run(
            args.models,
            args.data,
            args.results / "raw",
            args.cache,
            task_names=args.tasks,
            limit=args.limit,
            pull=args.pull,
            rm_after=args.rm_after,
            host=args.host,
        )
    elif args.bench_cmd == "report":
        md = report.write_report(args.results / "raw", args.results, Path("README.md"))
        console.print(md)


def _ingest(args: argparse.Namespace) -> None:
    from .rag.ingest import load_documents
    from .rag.store import LankaIndex

    docs, rep = load_documents(args.paths, args.chunk_size, args.chunk_overlap)
    console.print(
        f"{rep.files} files, {rep.pages} pages -> {rep.chunks} chunks  languages={rep.languages}"
    )
    if rep.skipped_legacy:
        console.print(
            f"[yellow]Skipped {len(rep.skipped_legacy)} page(s) that look like legacy-font "
            f"Sinhala (needs conversion or OCR):[/] {', '.join(rep.skipped_legacy[:10])}"
        )
    if not docs:
        raise SystemExit("Nothing to index.")
    with console.status(f"Embedding {len(docs)} chunks with {args.embedding}..."):
        index = LankaIndex.build(docs, args.embedding, args.host)
    index.save(args.index)
    console.print(f"[green]Index saved to {args.index}[/]")


def _print_answer(ans) -> None:
    console.print(f"\n[bold]{ans.answer}[/]\n")
    for i, d in enumerate(ans.sources, start=1):
        src = Path(str(d.metadata.get("source"))).name
        page = d.metadata.get("page")
        snippet = d.page_content[:140].replace("\n", " ")
        console.print(f"[dim][{i}] {src}{f' p.{page}' if page else ''}: {snippet}...[/]")


def _qa(args: argparse.Namespace) -> None:
    from .rag.chain import ask, build_chain, make_llm
    from .rag.store import LankaIndex

    index = LankaIndex.load(args.index, args.host)
    chain = build_chain(index.retriever(k=args.k, mode=args.mode), make_llm(args.llm, args.host))
    if args.cmd == "ask":
        _print_answer(ask(chain, args.question))
        return
    console.print("Ask in Sinhala, Tamil or English. Empty line to quit.")
    while q := console.input("[bold cyan]> [/]").strip():
        _print_answer(ask(chain, q))


def _inspect(args: argparse.Namespace) -> None:
    from .rag.ingest import load_pages
    from .text import legacy_font_score, normalize, script_stats

    t = Table("page", "chars", "Sinhala", "Tamil", "Latin", "legacy-font score", "verdict")
    for page, raw in load_pages(args.file):
        text = normalize(raw)
        s = script_stats(text)
        score = legacy_font_score(text)
        verdict = "legacy font: convert/OCR" if score >= 0.3 else (s.dominant if text else "empty")
        t.add_row(str(page or "-"), str(len(text)), str(s.sinhala), str(s.tamil), str(s.latin),
                  f"{score:.2f}", verdict)  # fmt: skip
    console.print(t)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="lanka-rag", description=__doc__)
    p.add_argument("--host", default=DEFAULT_OLLAMA_HOST, help="Ollama URL")
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("bench", help="embedding benchmark")
    bsub = b.add_subparsers(dest="bench_cmd", required=True)
    d = bsub.add_parser("download", help="download Belebele si/si-Latn/ta/en")
    d.add_argument("--data", type=Path, default=DATA_DIR)
    bsub.add_parser("models", help="list known models and their prompts")
    r = bsub.add_parser("run", help="score models on the retrieval tasks")
    r.add_argument("models", nargs="+", choices=list(MODELS), metavar="MODEL")
    r.add_argument("--tasks", nargs="+", help="subset of task names, e.g. 'si→si'")
    r.add_argument("--limit", type=int, help="only the first N questions (smoke test)")
    r.add_argument("--pull", action="store_true", help="ollama pull missing models")
    r.add_argument("--rm-after", action="store_true", help="ollama rm models pulled by this run")
    r.add_argument("--data", type=Path, default=DATA_DIR)
    r.add_argument("--results", type=Path, default=RESULTS_DIR)
    r.add_argument("--cache", type=Path, default=CACHE_DIR)
    rep = bsub.add_parser("report", help="write results/*.md, heatmaps, README table")
    rep.add_argument("--results", type=Path, default=RESULTS_DIR)

    i = sub.add_parser("ingest", help="index PDFs, text, markdown or HTML")
    i.add_argument("paths", nargs="+", type=Path)
    i.add_argument("--index", type=Path, default=INDEX_DIR)
    i.add_argument("--embedding", default=DEFAULT_EMBEDDING, choices=[
        k for k, m in MODELS.items() if m.backend == "ollama"])  # fmt: skip
    i.add_argument("--chunk-size", type=int, default=800)
    i.add_argument("--chunk-overlap", type=int, default=120)

    from .rag.chain import DEFAULT_LLM

    for name in ("ask", "chat"):
        q = sub.add_parser(name, help="ask one question" if name == "ask" else "interactive chat")
        if name == "ask":
            q.add_argument("question")
        q.add_argument("--index", type=Path, default=INDEX_DIR)
        q.add_argument("--llm", default=DEFAULT_LLM)
        q.add_argument("--k", type=int, default=4)
        q.add_argument("--mode", choices=["hybrid", "dense", "bm25"], default="hybrid")

    ins = sub.add_parser("inspect", help="per-page script stats and legacy-font detection")
    ins.add_argument("file", type=Path)
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    {"bench": _bench, "ingest": _ingest, "ask": _qa, "chat": _qa, "inspect": _inspect}[args.cmd](
        args
    )


if __name__ == "__main__":
    main()
