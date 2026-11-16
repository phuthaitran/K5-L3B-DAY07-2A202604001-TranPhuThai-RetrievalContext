"""Compare retrieval methods on one split with identical corpus, questions and labels.

Metrics (denominator = answerable questions of the split):
  Hit@1, Evidence Hit@3  - a hit = chunk with the gold source_id + page that contains the
                           gold evidence verbatim (starter `baseline.hit`).
  MRR@10                 - reciprocal rank of the first hit within the top 10.
  Context Evidence Rate  - gold evidence present in the packed context (starter
                           `pack_context`, char budget incl. the [id | source | page] header).

Queries are the original benchmark questions (no rewriting). `filter` is only the
user-provided scope stored with the question. Gold is used only for scoring.

Usage:
  python scripts/benchmark_retrieval.py --split dev --budgets 800 2500
"""
import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "starter"))

from dotenv import load_dotenv  # noqa: E402

from baseline import hit, pack_context  # noqa: E402
from src.retrievers import DenseRetriever, RetrieverError, build_retriever  # noqa: E402

RANK_DEPTH = 10  # retrieve 10 for MRR; Hit@1/@3 and context use the top 3
TOP_K = 3


def evaluate(name, retriever, questions, budgets):
    traces, errors = [], 0
    for q in questions:
        t0 = time.perf_counter()
        try:
            ranked = retriever.search(q["question"], filters=q.get("filter"), top_k=RANK_DEPTH)
            # Neighbour-expanded pipelines: rank on base hits only; context = top-3 hits + their neighbours.
            expanded = any("neighbor_of" in r for r in ranked)
            ranked = [r for r in ranked if "neighbor_of" not in r]
            top = (retriever.search(q["question"], filters=q.get("filter"), top_k=TOP_K)
                   if expanded or hasattr(retriever, "base") else ranked[:TOP_K])
        except RetrieverError as exc:
            errors += 1
            traces.append({"query_id": q["id"], "method": name, "status": "error", "error": str(exc)})
            continue
        latency_ms = (time.perf_counter() - t0) * 1000
        hits = [hit(r, q["gold"]) for r in ranked]
        first = next((i + 1 for i, h in enumerate(hits) if h), None)
        contexts = {}
        for budget in budgets:
            context, ids = pack_context(top, budget=budget)
            contexts[str(budget)] = {
                "context_ids": ids,
                "context_chars": len(context),
                "context_hit": any(hit(r, q["gold"]) and r["id"] in ids for r in top),
            }
        traces.append({
            "query_id": q["id"],
            "method": name,
            "status": "ok",
            "query": q["question"],
            "filter": q.get("filter"),
            "latency_ms": round(latency_ms, 2),
            "hit_at_1": bool(hits and hits[0]),
            "evidence_hit_at_3": any(hits[:TOP_K]),
            "first_hit_rank": first,
            "results": [
                {"rank": r["rank"], "id": r["id"], "score": round(r["score"], 6), "source": r["source"],
                 "page": r["page"], "section": r["metadata"].get("section"), "hit": h,
                 **({"source_ranks": r["source_ranks"]} if "source_ranks" in r else {})}
                for r, h in zip(ranked, hits)
            ],
            "contexts": contexts,
        })

    ok = [t for t in traces if t["status"] == "ok"]
    n = len(questions)

    def rate(k):
        return {"n": k, "N": n, "rate": round(k / n, 4) if n else None}

    summary = {
        "method": name,
        "split": questions[0]["split"] if questions else None,
        "N": n,
        "errors": errors,
        "top_k": TOP_K,
        "hit_at_1": rate(sum(t["hit_at_1"] for t in ok)),
        "evidence_hit_at_3": rate(sum(t["evidence_hit_at_3"] for t in ok)),
        "mrr_at_10": round(sum(1 / t["first_hit_rank"] for t in ok if t["first_hit_rank"]) / n, 4) if n else None,
        "context": {
            str(b): {
                "context_evidence_rate": rate(sum(t["contexts"][str(b)]["context_hit"] for t in ok)),
                "avg_context_chars": round(statistics.mean(t["contexts"][str(b)]["context_chars"] for t in ok), 1) if ok else None,
                "avg_chunks_in_context": round(statistics.mean(len(t["contexts"][str(b)]["context_ids"]) for t in ok), 2) if ok else None,
            }
            for b in budgets
        },
        "median_latency_ms": round(statistics.median(t["latency_ms"] for t in ok), 2) if ok else None,
        "cost_usd": 0.0,  # all retrieval runs locally (CPU embedding + local Postgres)
    }
    return summary, traces


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", choices=("dev", "test"), default="dev")
    parser.add_argument("--budgets", type=int, nargs="+", default=[800, 2500, 5000])
    parser.add_argument("--methods", nargs="+", default=["tfidf", "bm25", "dense", "hybrid", "hybrid_nb"])
    parser.add_argument("--chunks", type=Path, default=ROOT / "data/chunks.json")
    parser.add_argument("--questions", type=Path, default=ROOT / "data/questions.json")
    parser.add_argument("--output", type=Path, default=ROOT / "results/retrieval")
    args = parser.parse_args()

    load_dotenv(ROOT / ".env")
    chunks = json.loads(args.chunks.read_text(encoding="utf-8"))
    questions = [q for q in json.loads(args.questions.read_text(encoding="utf-8")) if q["split"] == args.split]
    print(f"{len(chunks)} chunks, {len(questions)} answerable {args.split} questions, budgets={args.budgets}")

    dense = DenseRetriever() if {"dense", "hybrid"} & set(args.methods) else None
    summaries, all_traces = [], []
    for name in args.methods:
        summary, traces = evaluate(name, build_retriever(name, chunks, dense=dense), questions, args.budgets)
        summaries.append(summary)
        all_traces.extend(traces)

    header = f"{'method':<8} {'Hit@1':>11} {'EvHit@3':>11} {'MRR@10':>7} " + " ".join(
        f"{'Ctx@' + str(b):>11}" for b in args.budgets) + f" {'med ms':>7} {'err':>4}"
    print(header)
    print("-" * len(header))
    for s in summaries:
        fmt = lambda r: f"{r['n']}/{r['N']} {r['rate']:.0%}"  # noqa: E731
        ctx = " ".join(f"{fmt(s['context'][str(b)]['context_evidence_rate']):>11}" for b in args.budgets)
        print(f"{s['method']:<8} {fmt(s['hit_at_1']):>11} {fmt(s['evidence_hit_at_3']):>11} "
              f"{s['mrr_at_10']:>7.3f} {ctx} {s['median_latency_ms']:>7} {s['errors']:>4}")

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / f"summary_{args.split}.json").write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.output / f"traces_{args.split}.json").write_text(
        json.dumps(all_traces, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"saved to {args.output}")


if __name__ == "__main__":
    main()
