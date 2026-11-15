"""Download a demo corpus of Sri Lankan Acts in Sinhala (text only, ~1,000 pages).

Source: avishadilhara/sinhala-ocr-lk-acts-1010 on Hugging Face (CC BY 4.0),
hand-corrected transcriptions of scanned Acts from the 1980s-2010s. We fetch only
the text column through the Dataset Viewer API, not the page images.

    uv run python scripts/fetch_acts_demo.py            # -> docs/acts/*.txt
    uv run lanka-rag ingest docs/acts
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import httpx

API = "https://datasets-server.huggingface.co/rows"
DATASET = "avishadilhara/sinhala-ocr-lk-acts-1010"
SPLITS = {"train": 707, "eval": 101, "test": 202}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("docs/acts"))
    ap.add_argument("--splits", nargs="+", default=list(SPLITS), choices=list(SPLITS))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    n = 0
    with httpx.Client(timeout=120) as client:
        for split in args.splits:
            for offset in range(0, SPLITS[split], 100):
                params = {
                    "dataset": DATASET,
                    "config": "default",
                    "split": split,
                    "offset": offset,
                    "length": 100,
                }
                for attempt in range(5):  # the Dataset Viewer API sometimes returns 502
                    r = client.get(API, params=params)
                    if r.status_code < 500:
                        break
                    time.sleep(2**attempt)
                r.raise_for_status()
                for item in r.json()["rows"]:
                    row = item["row"]
                    path = args.out / f"{row['filename']}.txt"
                    path.write_text(row["text"], encoding="utf-8")
                    n += 1
    print(f"Wrote {n} pages to {args.out} (CC BY 4.0, {DATASET})")


if __name__ == "__main__":
    main()
