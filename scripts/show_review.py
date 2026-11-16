"""Print a run side by side with the reference answer, gold evidence and cited chunk text,
to support manual labelling (reviews.json). Gold is shown to the grader only.

Usage: python scripts/show_review.py results/agent/<config>-<split> [--ids q01 q07]
"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--ids", nargs="*")
    args = parser.parse_args()
    load = lambda p: json.loads(p.read_text(encoding="utf-8"))  # noqa: E731
    traces = load(args.run_dir / "traces.json")
    chunks = {c["id"]: c for c in load(ROOT / "data/chunks.json")}
    cases = {c["id"]: c for c in load(ROOT / "data/agent_cases.json")}
    gold = {q["id"]: q["gold"] for q in load(ROOT / "data/questions.json")}
    for t in traces:
        qid = t["query_id"]
        if args.ids and qid not in args.ids:
            continue
        c = cases[qid]
        print(f"===== {qid} [{t['status']}] answerable={c['answerable']} "
              f"ctx_evidence={t.get('final_context_evidence_hit')} searches={t['searches']}")
        print("Q:  ", c["question"])
        print("REF:", c.get("expected_answer") or c.get("absence_reason"))
        for g in gold.get(qid, []):
            print(f"GOLD p{g['page']}:", g["evidence"].replace("\n", " ")[:300])
        for e in t["events"]:
            if e["event"] == "search":
                print("  search:", e["query"], "->", [f"{r['alias']}/p{r['page']}" for r in e["results"]])
            elif e["event"] in ("abstain", "invalid_or_missing_citations", "error", "search_limit"):
                print("  ", {k: v for k, v in e.items() if k != "event"} | {"event": e["event"]})
        print("ANS:", t.get("raw_answer") or t["answer"] or t.get("error"))
        for i in t["citations"]:
            print(f"  CITED {i[:8]} p{chunks[i]['page']}:", chunks[i]["text"].replace("\n", " ")[:700])
        print()


if __name__ == "__main__":
    main()
