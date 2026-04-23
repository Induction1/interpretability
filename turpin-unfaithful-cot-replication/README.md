# Turpin “always (A)” BBH replication

Small replication of the **positional / always-(A) few-shot** bias on Big-Bench Hard (BBH), from *[Language Models Don’t Always Say What They Think](https://arxiv.org/abs/2305.04388)* (Turpin et al., NeurIPS 2023). The workflow lives in **`bbh_replication.ipynb`**.

## What this found

Tested **Claude Haiku** on 3 BBH tasks (n = 50 each). When few-shot examples are rigged so the correct answer is always **(A)**, accuracy drops **6 pp** on both `causal_judgement` and `sports_understanding`. The bias had no effect on `ruin_names` — a subjective humor task where no option is systematically more plausible-sounding. The directional finding from the paper replicates clearly on a model two generations newer.

The mechanism also replicates: in harmed examples the model’s CoT reaches the correct conclusion, then the final line overrides it to match the biased option. Stated reasoning does not reflect what determined the answer.

## Setup

- **Python 3.11** (see `pyproject.toml`).
- **[PDM](https://pdm-project.org/)** for dependencies and the project venv.

```bash
cd turpin-unfaithful-cot-replication
pdm install
```

Create **`.env`** in this folder with your Anthropic API key:

```bash
ANTHROPIC_API_KEY=sk-ant-...
```

Do **not** commit `.env`. API calls cost money (~$1–2 for the full 300-call run on Haiku).

## Running

```bash
pdm run jupyter notebook
```

Open **`bbh_replication.ipynb`** from this directory (so paths and `load_dotenv` resolve). Run cells top to bottom. The batch cell skips automatically when the JSONL cache is complete — set `RUN_API_BATCH = True` only when you want to (re)run the API calls.

## Data

BBH task JSON (val slice + few-shot prefixes) lives under **`data/bbh/`**. See **`data/bbh/NOTICE`** for provenance and license.

## Results

Cached completions are written under **`results/`** (ignored by git). Regenerate by running the batch step in the notebook.

## License

MIT (see `pyproject.toml`).
