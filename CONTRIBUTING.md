# Contributing

Thanks for helping make Sinhala and Tamil work better in local AI tools. Issues and pull
requests in English, Sinhala or Tamil are all fine.

## Setup

```bash
git clone https://github.com/Nipuna-Lakruwan/lanka-rag.git
cd lanka-rag
uv sync
```

Before opening a pull request, run the same checks as CI:

```bash
uv run ruff check . && uv run ruff format --check . && uv run pytest -q
```

## Benchmark a new embedding model

This is the most useful contribution: one new row in the results table.

1. Add an `EmbeddingModel(...)` entry to `MODELS` in
   [`src/lanka_rag/models.py`](src/lanka_rag/models.py). Use the query and document prefixes
   from the model card. Wrong prefixes can cost 10+ nDCG points, so link the card in `notes`.
2. Run it and regenerate the report:

   ```bash
   uv run lanka-rag bench download
   uv run lanka-rag bench run <model-key> --pull
   uv run lanka-rag bench report
   ```

3. Commit `results/raw/<model-key>.json` together with the regenerated `results/results.md`,
   `results/summary.json`, both heatmaps and the README results section. Mention your
   hardware in the PR, because the speed column depends on it.

Hugging Face models (`backend="hf"`) need `uv sync --extra hf`.

## Add Sinhala or Tamil test data

Retrieval sets built from real Sri Lankan documents (gazettes, Acts, news, university
material) are very welcome, especially with questions written by native speakers. Please:

- include only text whose license allows redistribution, and state the license;
- keep Sinhala and Tamil in Unicode (`lanka-rag inspect <file>` flags legacy-font text);
- open an issue first, so we can agree on the format before you spend time on it.

## Reporting a problem with Sinhala text

If something looks wrong (broken conjuncts, missing vowel signs, garbled PDF text), attach a
small sample file or the exact string, plus the output of `lanka-rag inspect <file>`.
