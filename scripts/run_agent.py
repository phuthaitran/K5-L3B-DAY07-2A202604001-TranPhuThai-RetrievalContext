"""Run the Gemini agent on one split and save traces + a manual review sheet.

Only `question` and the user-provided `filter` of each case reach the agent. Gold
evidence / expected answers are used afterwards for scoring the final context and are
copied into the review sheet for the human grader.

Outputs in <output>/<config-name>-<split>/:
  config.json        - agent + retriever + model configuration
  traces.json        - one record per case (query_id, status, answer, citations, events, usage...)
                       compatible with starter/agent_metrics.py
  run_summary.json   - automatic metrics: status counts, tool use, searches, context evidence
                       rate, latency, tokens, estimated cost
  review_sheet.json  - to be labelled by hand (answer_grade, citation_support, abstention_correct)

Usage:
  python scripts/run_agent.py --split dev --name hybrid_k3_b2500_s3 --retriever hybrid --top-k 3 --budget 2500 --max-searches 3
  python scripts/run_agent.py --split dev --ids q01 q_dev_unabs_01   # quick smoke run
"""
import argparse
import json
import os
import statistics
import sys
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "starter"))

from dotenv import load_dotenv  # noqa: E402

from baseline import hit  # noqa: E402
from src.agent import AgentConfig, GeminiController, run_agent  # noqa: E402
from src.retrievers import build_retriever  # noqa: E402


def estimate_cost(usage):
    pin = os.environ.get("GEMINI_PRICE_INPUT_PER_MTOK")
    pout = os.environ.get("GEMINI_PRICE_OUTPUT_PER_MTOK")
    if not pin or not pout:
        return None
    return (usage["prompt_tokens"] * float(pin)
            + (usage["output_tokens"] + usage["thinking_tokens"]) * float(pout)) / 1e6


def summarize(traces, cases, gold_by_id, chunks_by_id):
    ok = [t for t in traces if t["status"] != "error"]
    answerable = {c["id"] for c in cases if c["answerable"]}
    ctx_hits = []
    for t in ok:
        if t["query_id"] in answerable:
            gold = gold_by_id[t["query_id"]]
            t["final_context_evidence_hit"] = any(hit(chunks_by_id[i], gold) for i in t["final_context_ids"])
            t["retrieved_evidence_hit"] = any(
                hit(chunks_by_id[r["id"]], gold) for e in t["events"] if e["event"] == "search" for r in e["results"])
            ctx_hits.append(t["final_context_evidence_hit"])
    tot = {k: sum(t["usage"][k] for t in traces) for k in ("prompt_tokens", "output_tokens", "thinking_tokens", "llm_calls")}
    n_ans = sum(1 for t in ok if t["query_id"] in answerable)
    return {
        "cases": len(traces),
        "errors": len(traces) - len(ok),
        "status": dict(Counter(t["status"] for t in traces)),
        "tool_use": {"n": sum(t["searches"] > 0 for t in ok), "N": len(ok)},
        "searches_distribution": dict(sorted(Counter(t["searches"] for t in ok).items())),
        "answerable_answered": {"n": sum(t["status"] == "answered" for t in ok if t["query_id"] in answerable), "N": n_ans},
        "unanswerable_abstained": {"n": sum(t["status"] == "abstain" for t in ok if t["query_id"] not in answerable),
                                   "N": len(ok) - n_ans},
        "context_evidence_rate": {"n": sum(ctx_hits), "N": len(ctx_hits)},
        "retrieved_evidence_rate": {"n": sum(t.get("retrieved_evidence_hit", False) for t in ok if t["query_id"] in answerable),
                                    "N": n_ans},
        "avg_final_context_chars": round(statistics.mean(t["final_context_chars"] for t in ok), 1) if ok else None,
        "median_latency_s": round(statistics.median(t["latency_ms"] for t in ok) / 1000, 2) if ok else None,
        "total_latency_s": round(sum(t["latency_ms"] for t in traces) / 1000, 1),
        "tokens": tot,
        "estimated_cost_usd": estimate_cost(tot),
    }


def review_sheet(traces, cases_by_id, chunks_by_id):
    sheet = []
    for t in traces:
        if t["status"] == "error":
            continue
        c = cases_by_id[t["query_id"]]
        row = {
            "query_id": t["query_id"],
            "answerable": c["answerable"],
            "question": c["question"],
            "reference": c.get("expected_answer") or c.get("absence_reason"),
            "status": t["status"],
            "answer": t["answer"],
            "raw_answer": t.get("raw_answer"),
            "abstain_reason": t.get("abstain_reason"),
            "cited_chunks": [{"id": i, "page": chunks_by_id[i]["page"], "text": chunks_by_id[i]["text"]}
                             for i in t["citations"]],
            "citation_support": [None] * len(t["citations"]),
            "notes": "",
        }
        if c["answerable"]:
            row["answer_grade"] = 0 if t["status"] == "abstain" else None
        else:
            row["abstention_correct"] = None
        sheet.append(row)
    return sheet


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", choices=("dev", "test"), required=True)
    parser.add_argument("--name", default=None, help="config name (default derived from params)")
    parser.add_argument("--retriever", choices=("dense", "bm25", "hybrid", "tfidf", "dense_nb", "bm25_nb", "hybrid_nb"), default="hybrid")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--budget", type=int, default=2500, help="context budget in characters")
    parser.add_argument("--max-searches", type=int, default=3)
    parser.add_argument("--context-order", choices=("append", "rank"), default="append")
    parser.add_argument("--thinking-budget", type=int, default=None)
    parser.add_argument("--ids", nargs="*", help="run only these case IDs (smoke test)")
    parser.add_argument("--sleep", type=float, default=0.0, help="seconds between cases (rate limits)")
    parser.add_argument("--output", type=Path, default=ROOT / "results/agent")
    args = parser.parse_args()

    load_dotenv(ROOT / ".env")
    cfg = AgentConfig(
        name=args.name or f"{args.retriever}_k{args.top_k}_b{args.budget}_s{args.max_searches}"
                          + ("" if args.context_order == "append" else f"_{args.context_order}"),
        retriever=args.retriever, top_k=args.top_k, budget=args.budget,
        max_searches=args.max_searches, context_order=args.context_order,
    )
    chunks = json.loads((ROOT / "data/chunks.json").read_text(encoding="utf-8"))
    chunks_by_id = {c["id"]: c for c in chunks}
    cases = [c for c in json.loads((ROOT / "data/agent_cases.json").read_text(encoding="utf-8"))
             if c["split"] == args.split and (not args.ids or c["id"] in args.ids)]
    gold_by_id = {q["id"]: q["gold"] for q in json.loads((ROOT / "data/questions.json").read_text(encoding="utf-8"))}

    retriever = build_retriever(cfg.retriever, chunks)
    controller = GeminiController(thinking_budget=args.thinking_budget)
    out_dir = args.output / f"{cfg.name}-{args.split}" if not args.ids else args.output / f"smoke-{cfg.name}-{args.split}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps({
        **asdict(cfg), "split": args.split, "model": controller.model, "temperature": controller.temperature,
        "thinking_budget": args.thinking_budget, "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        "vector_store": "PostgreSQL + pgvector (Day 6, dsa_chunks, strategy B)", "case_ids": [c["id"] for c in cases],
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }, indent=2) + "\n", encoding="utf-8")

    traces = []
    for i, case in enumerate(cases, 1):
        result = run_agent(case["question"], controller, retriever, cfg, filters=case.get("filter"))
        traces.append({"query_id": case["id"], "split": case["split"], "question": case["question"],
                       "config": cfg.name, "model": controller.model, **result})
        print(f"[{i}/{len(cases)}] {case['id']:<15} {result['status']:<9} searches={result['searches']} "
              f"calls={result['usage']['llm_calls']} {result['latency_ms'] / 1000:.1f}s "
              f"{(result.get('error') or result['answer'])[:90]!r}")
        (out_dir / "traces.json").write_text(json.dumps(traces, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        if args.sleep:
            time.sleep(args.sleep)

    summary = {"config": cfg.name, "split": args.split, "model": controller.model,
               **summarize(traces, cases, gold_by_id, chunks_by_id)}
    (out_dir / "traces.json").write_text(json.dumps(traces, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out_dir / "run_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out_dir / "review_sheet.json").write_text(
        json.dumps(review_sheet(traces, {c["id"]: c for c in cases}, chunks_by_id), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"saved to {out_dir}")


if __name__ == "__main__":
    main()
