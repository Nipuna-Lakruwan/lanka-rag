"""Turn per-model result JSON into markdown tables and an nDCG heatmap (SVG)."""

from __future__ import annotations

import json
from html import escape
from pathlib import Path

from .datasets import TASKS

# Sequential blue ramp, light -> dark (steps 100..700).
BLUE = [
    "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
    "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b",
]  # fmt: skip
THEMES = {
    "light": {"surface": "#fcfcfb", "ink": "#0b0b0b", "ink2": "#52514e", "ramp": BLUE,
              "on_dark": "#fcfcfb", "on_light": "#0b0b0b"},
    # On a dark surface "near zero" must recede into the background, so the ramp flips.
    "dark": {"surface": "#1a1a19", "ink": "#f0efec", "ink2": "#c3c2b7", "ramp": BLUE[::-1],
             "on_dark": "#f0efec", "on_light": "#0b0b0b"},
}  # fmt: skip

SINHALA_TASKS = ["si→si", "si→si (no ZWJ)", "si-Latn→si", "en→si", "si→en", "ta→si", "si→ta"]
START, END = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"


def load_results(raw_dir: Path) -> list[dict]:
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted(raw_dir.glob("*.json"))]


def _metric(rec: dict, task: str, metric: str) -> float | None:
    for r in rec["results"]:
        if r["task"] == task:
            return r["metrics"][metric]
    return None


def _avg(rec: dict, tasks: list[str], metric: str = "ndcg@10") -> float:
    vals = [v for t in tasks if (v := _metric(rec, t, metric)) is not None]
    return sum(vals) / len(vals) if vals else 0.0


def sort_records(records: list[dict]) -> list[dict]:
    return sorted(records, key=lambda r: _avg(r, SINHALA_TASKS), reverse=True)


def task_names(records: list[dict]) -> list[str]:
    present = {r["task"] for rec in records for r in rec["results"]}
    return [t.name for t in TASKS if t.name in present]


def markdown_tables(records: list[dict]) -> str:
    records = sort_records(records)
    tasks = task_names(records)
    lines = [
        "**nDCG@10** (higher is better; best per column in bold). "
        '"Sinhala avg" averages every task with Sinhala on either side.',
        "",
        "| Model | Params | Sinhala avg | " + " | ".join(tasks) + " |",
        "|---|---:|---:|" + "---:|" * len(tasks),
    ]
    best = {t: max((_metric(r, t, "ndcg@10") or 0) for r in records) for t in tasks}
    for rec in records:
        m = rec["model"]
        cells = []
        for t in tasks:
            v = _metric(rec, t, "ndcg@10")
            s = "-" if v is None else f"{v:.3f}"
            cells.append(f"**{s}**" if v is not None and v == best[t] else s)
        name = m["key"] + ("" if m["multilingual"] else " †")
        lines.append(
            f"| {name} | {m['params']} | {_avg(rec, SINHALA_TASKS):.3f} | "
            + " | ".join(cells)
            + " |"
        )
    lines += ["", "† English-centric model, included as a control.", ""]

    lines += [
        "**Recall@1 / Recall@10** on the three monolingual tasks, plus 95% bootstrap CI of nDCG@10 "
        "for si→si:",
        "",
        "| Model | si→si R@1 | si→si R@10 | si→si nDCG@10 95% CI | ta→ta R@1 | Singlish→si R@1 |"
        " Embed speed (passages/s) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for rec in records:
        si = next((r for r in rec["results"] if r["task"] == "si→si"), None)
        lo, hi = si["ndcg@10_ci95"] if si else (0, 0)
        speed = si["timing"].get("docs_per_second") if si else None

        def f(task: str, metric: str, rec: dict = rec) -> str:
            v = _metric(rec, task, metric)
            return "-" if v is None else f"{v:.3f}"

        lines.append(
            f"| {rec['model']['key']} | {f('si→si', 'recall@1')} | {f('si→si', 'recall@10')} | "
            f"{lo:.3f}–{hi:.3f} | {f('ta→ta', 'recall@1')} | {f('si-Latn→si', 'recall@1')} | "
            f"{speed if speed else '-'} |"
        )
    return "\n".join(lines) + "\n"


def heatmap_svg(records: list[dict], theme: str = "light") -> str:
    th = THEMES[theme]
    records = sort_records(records)
    tasks = task_names(records)
    cell_w, cell_h, gap = 78, 30, 2
    left, top, right, bottom = 200, 74, 16, 44
    width = left + len(tasks) * (cell_w + gap) + right
    height = top + len(records) * (cell_h + gap) + bottom
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'width="{width}" height="{height}" font-family="system-ui,-apple-system,Segoe UI,'
        f'Noto Sans Sinhala,Noto Sans Tamil,sans-serif" role="img" '
        f'aria-label="Heatmap of nDCG@10 per model and retrieval task">',
        f'<rect width="100%" height="100%" fill="{th["surface"]}"/>',
        f'<text x="16" y="26" font-size="15" font-weight="600" fill="{th["ink"]}">'
        "Sinhala / Tamil retrieval on Belebele: nDCG@10</text>",
        f'<text x="16" y="45" font-size="12" fill="{th["ink2"]}">query language → passage '
        f"language · 900 questions × 488 passages per task · "
        f"{'darker' if theme == 'light' else 'brighter'} = better</text>",
    ]
    for j, t in enumerate(tasks):
        x = left + j * (cell_w + gap) + cell_w / 2
        out.append(
            f'<text x="{x}" y="{top - 8}" font-size="11" text-anchor="middle" '
            f'fill="{th["ink2"]}">{escape(t)}</text>'
        )
    ramp = th["ramp"]
    for i, rec in enumerate(records):
        y = top + i * (cell_h + gap)
        out.append(
            f'<text x="{left - 10}" y="{y + cell_h / 2 + 4}" font-size="12" text-anchor="end" '
            f'fill="{th["ink"]}">{escape(rec["model"]["key"])}</text>'
        )
        for j, t in enumerate(tasks):
            v = _metric(rec, t, "ndcg@10")
            x = left + j * (cell_w + gap)
            if v is None:
                continue
            step = min(len(ramp) - 1, int(round(v * (len(BLUE) - 1))))
            fill = ramp[step]
            # Ink by luminance of the cell (BLUE index: >= 6 is dark enough for light ink).
            dark_cell = step >= 6 if theme == "light" else step < 6
            ink = th["on_dark"] if dark_cell else th["on_light"]
            out.append(
                f'<rect x="{x}" y="{y}" width="{cell_w}" height="{cell_h}" rx="4" fill="{fill}">'
                f"<title>{escape(rec['model']['key'])} · {escape(t)}: {v:.3f}</title></rect>"
            )
            out.append(
                f'<text x="{x + cell_w / 2}" y="{y + cell_h / 2 + 4}" font-size="12" '
                f'text-anchor="middle" fill="{ink}">{v:.2f}</text>'
            )
    note_y = height - 16
    out.append(
        f'<text x="16" y="{note_y}" font-size="11" fill="{th["ink2"]}">Rows sorted by mean '
        "over Sinhala tasks. Local Ollama models on an Apple M1; see README for setup.</text>"
    )
    out.append("</svg>")
    return "\n".join(out)


def write_report(raw_dir: Path, out_dir: Path, readme: Path | None = None) -> str:
    records = load_results(raw_dir)
    if not records:
        raise SystemExit(f"No results in {raw_dir}. Run `lanka-rag bench run` first.")
    md = markdown_tables(records)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.md").write_text(md, encoding="utf-8")
    for theme in THEMES:
        (out_dir / f"ndcg_heatmap_{theme}.svg").write_text(
            heatmap_svg(records, theme), encoding="utf-8"
        )
    summary = [
        {
            "model": r["model"]["key"],
            "params": r["model"]["params"],
            "sinhala_avg_ndcg@10": round(_avg(r, SINHALA_TASKS), 4),
            "tasks": {x["task"]: x["metrics"] for x in r["results"]},
        }
        for r in sort_records(records)
    ]
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    if readme and readme.exists():
        text = readme.read_text(encoding="utf-8")
        if START in text and END in text:
            head, rest = text.split(START, 1)
            _, tail = rest.split(END, 1)
            readme.write_text(f"{head}{START}\n{md}{END}{tail}", encoding="utf-8")
    return md
