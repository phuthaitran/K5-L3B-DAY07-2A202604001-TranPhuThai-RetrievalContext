"""Export Day 6 chunks from pgvector and build the Day 7 benchmark files.

Outputs (all JSON arrays, UTF-8):
  data/chunks.json       - Strategy B chunks exported from the Day 6 `dsa_chunks` table
  data/questions.json    - 20 answerable questions (15 dev / 5 test) with gold evidence
  data/agent_cases.json  - 24 agent cases = 20 answerable + 4 unanswerable (2 dev / 2 test)

Usage:
  python scripts/prepare_data.py --day6-benchmark <path/to/day6/starter/data/dsa_benchmark.json>

DATABASE_URL is read from the environment / .env (see .env.example).

Notes on gold evidence:
  * Gold is taken unchanged from the Day 6 benchmark, except q02 and q11 whose
    `evidence_quote` in Day 6 collapsed the paragraph/bullet line breaks. The text
    is identical but whitespace differed, so a verbatim substring check failed. We
    replace those two quotes with the exact chunk text (same words, original line
    breaks). Every evidence is then verified to sit fully inside one chunk with the
    same source_id and page; the script fails otherwise.
  * Unanswerable questions are deliberately close to the document topic (hashing,
    shortest paths, Turing machines, merge sort). Absence is verified by keyword
    checks over the whole corpus (not by top-k retrieval); see ABSENCE_CHECKS.
"""
import argparse
import json
import os
from pathlib import Path

import psycopg
from dotenv import load_dotenv

DATA_DIR = Path("data")
STRATEGY = "B"

WHITESPACE_FIXED_EVIDENCE = {
    "q02": (
        "Computer scientists apply a top-down approach.\n\n"
        "1. Select an appropriate abstract data type by analyzing the problem to determine the necessary functionality and operations.\n\n"
        "2. Select an appropriate data structure by quantifying the resource constraints (usually program running time) for each operation."
    ),
    "q11": (
        "• In the best-case situation (when the target word is at the start of the list), sequential search takes just one repetition to find the word in the list.\n\n"
        "• In the worst-case situation (when the target word is either at the end of the list or not in the list at all), sequential search takes N repetitions where N is the number of words in the list."
    ),
}

UNANSWERABLE = [
    {
        "id": "q_dev_unabs_01",
        "split": "dev",
        "question": "What load factor threshold does the text recommend before resizing a hash table to keep collisions rare?",
        "absence_reason": (
            "The hashing section (p.32) only states that hash tables must handle collisions; it never discusses "
            "load factor, resizing or rehashing."
        ),
        "absence_terms": ["load factor", "resiz", "rehash"],
    },
    {
        "id": "q_dev_unabs_02",
        "split": "dev",
        "question": "Which algorithm does the text recommend instead of Dijkstra's algorithm when a graph has negative edge weights?",
        "absence_reason": (
            "Dijkstra's algorithm is described (p.35), but the corpus never mentions negative edge weights "
            "or Bellman-Ford; the only occurrence of 'negative' is about integer data types (p.2)."
        ),
        "absence_terms": ["negative edge", "negative weight", "bellman"],
    },
    {
        "id": "q_test_unabs_01",
        "split": "test",
        "question": "Within Section 3.6, how does the text prove that the halting problem is undecidable?",
        "absence_reason": (
            "Section 3.6 covers Turing machines and complexity classes (P, NP, NP-complete) but never mentions "
            "the halting problem, decidability or undecidability."
        ),
        "absence_terms": ["halting", "undecid", "decidab"],
    },
    {
        "id": "q_test_unabs_02",
        "split": "test",
        "question": "According to the text, is merge sort a stable sorting algorithm, and why does stability matter?",
        "absence_reason": (
            "Merge sort is described (Section 3.5), but the corpus never discusses sorting stability or the "
            "relative order of equal keys."
        ),
        "absence_terms": ["stable sort", "stability", "relative order", "equal keys"],
    },
]


def export_chunks(database_url: str) -> list[dict]:
    with psycopg.connect(database_url) as conn:
        rows = conn.execute(
            """
            SELECT chunk_id, document_id, source, page, content, metadata
            FROM dsa_chunks
            WHERE strategy = %s
            ORDER BY page, chunk_id
            """,
            (STRATEGY,),
        ).fetchall()
    return [
        {
            "id": r[0],
            "source_id": r[1],
            "source": r[2],
            "page": r[3],
            "text": r[4],
            "metadata": r[5] if isinstance(r[5], dict) else {},
        }
        for r in rows
    ]


def build_answerable(day6_questions: list[dict], chunks: list[dict]) -> tuple[list[dict], list[dict]]:
    questions, cases = [], []
    for q in day6_questions:
        qid = q["id"]
        evidence = WHITESPACE_FIXED_EVIDENCE.get(qid, q["evidence_quote"])
        src, page = q["gold_source_id"], q["gold_page"]
        if not any(c["source_id"] == src and c["page"] == page and evidence in c["text"] for c in chunks):
            raise ValueError(f"{qid}: evidence not found verbatim in a chunk on page {page}")

        item = {
            "id": qid,
            "split": q["split"],
            "question": q["question"],
            "expected_answer": q["expected_answer"],
            "gold": [{"source_id": src, "page": page, "evidence": evidence}],
        }
        case = {
            "id": qid,
            "split": q["split"],
            "question": q["question"],
            "answerable": True,
            "expected_behavior": "search_and_answer",
            "expected_answer": q["expected_answer"],
        }
        # `filter` is user-provided scope from the Day 6 benchmark, not derived from gold.
        if q.get("filter"):
            item["filter"] = case["filter"] = q["filter"]
        questions.append(item)
        cases.append(case)
    return questions, cases


def build_unanswerable(chunks: list[dict]) -> list[dict]:
    corpus = "\n".join(c["text"] for c in chunks).casefold()
    cases = []
    for u in UNANSWERABLE:
        found = [t for t in u["absence_terms"] if t in corpus]
        if found:
            raise ValueError(f"{u['id']}: absence term(s) {found} occur in the corpus")
        cases.append({
            "id": u["id"],
            "split": u["split"],
            "question": u["question"],
            "answerable": False,
            "expected_behavior": "search_and_abstain",
            "absence_reason": u["absence_reason"],
            "absence_check": f"Case-insensitive search over all {len(chunks)} chunks: 0 matches for {u['absence_terms']}",
        })
    return cases


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--day6-benchmark", type=Path, required=True,
                        help="Path to Day 6 starter/data/dsa_benchmark.json")
    args = parser.parse_args()

    load_dotenv()
    database_url = os.environ["DATABASE_URL"]
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    chunks = export_chunks(database_url)
    print(f"Exported {len(chunks)} Strategy {STRATEGY} chunks from Day 6 pgvector")

    day6 = json.loads(args.day6_benchmark.read_text(encoding="utf-8"))["questions"]
    questions, cases = build_answerable(day6, chunks)
    cases += build_unanswerable(chunks)

    for name, obj in (("chunks.json", chunks), ("questions.json", questions), ("agent_cases.json", cases)):
        (DATA_DIR / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    for split in ("dev", "test"):
        ans = sum(c["split"] == split and c["answerable"] for c in cases)
        una = sum(c["split"] == split and not c["answerable"] for c in cases)
        print(f"{split}: {ans} answerable + {una} unanswerable = {ans + una}")
    print(f"total: {len(cases)} cases")


if __name__ == "__main__":
    main()
