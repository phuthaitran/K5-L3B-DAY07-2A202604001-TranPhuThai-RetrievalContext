# Code

| File | Purpose |
|---|---|
| `src/retrievers.py` | Dense (Day 6 pgvector), BM25, Hybrid (weighted RRF), TF-IDF wrapper; common result schema `id, text, source_id, source, page, metadata, score, rank, method` |
| `src/agent.py` | Gemini function-calling agent (`search` / `answer` / `abstain`), context packing, citation validation, traces |
| `scripts/prepare_data.py` | Export Day 6 chunks from pgvector, build `data/questions.json` and `data/agent_cases.json` |
| `scripts/benchmark_retrieval.py` | Compare retrieval methods on one split (Hit@1, Evidence Hit@3, MRR@10, Context Evidence Rate) |
| `scripts/run_agent.py` | Run the agent on one split; writes traces, run summary and a manual review sheet |

## Setup

```bash
python -m venv .venv && .venv/Scripts/activate      # Windows; use bin/activate on Linux/macOS
pip install -r requirements.txt
cp .env.example .env                                  # fill DATABASE_URL, GEMINI_API_KEY, GEMINI_MODEL
```

The Day 6 Postgres container (`pgvector/pgvector`, port 55433) must be running with table
`dsa_chunks` (Strategy B, 238 chunks, 384-dim `all-MiniLM-L6-v2` embeddings).

## Run

```bash
# 1. data (only needed to regenerate data/*.json from Day 6)
python scripts/prepare_data.py --day6-benchmark <day6-repo>/starter/data/dsa_benchmark.json

# 2. retrieval comparison on dev (gold is used only for scoring)
python scripts/benchmark_retrieval.py --split dev --budgets 800 2500
python starter/baseline.py --chunks data/chunks.json --questions data/questions.json --split dev --output results/baseline-tfidf-dev

# 3. agent on dev (real Gemini model + real vector store)
python scripts/run_agent.py --split dev --retriever hybrid --top-k 3 --budget 2500 --max-searches 3

# 4. after filling review_sheet.json by hand
python starter/agent_metrics.py --cases data/agent_cases.json --traces results/agent/<config>-dev/traces.json \
    --reviews results/agent/<config>-dev/review_sheet.json --split dev --output results/agent/<config>-dev/metrics.json

# tests (PYTHONUTF8=1 is needed on Windows for the starter baseline test)
$env:PYTHONUTF8 = "1"
python -m unittest discover -s tests -v
```

Outputs go to `results/` (tracked by Git). Context budget is measured in characters of the
packed context, passage headers included.
