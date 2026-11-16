"""Code-path tests for src/agent.py with a scripted controller (not a benchmark)."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.agent import AgentConfig, ContextState, Decision, ProviderError, run_agent  # noqa: E402

CHUNKS = [
    {"id": f"hash{i}", "text": f"text {i} " * 20, "source": "DSA.pdf", "source_id": "d", "page": i,
     "metadata": {"section": "3.1"}}
    for i in range(1, 7)
]


class FakeRetriever:
    def __init__(self):
        self.calls = 0

    def search(self, query, *, filters=None, top_k=3):
        start = self.calls * top_k
        self.calls += 1
        return [{**c, "score": 1.0 - r / 10, "rank": r + 1} for r, c in enumerate(CHUNKS[start:start + top_k])]


class ScriptedController:
    def __init__(self, actions):
        self.actions = list(actions)
        self.seen = []

    def decide(self, question, queries, context, searches_left):
        self.seen.append((list(queries), context, searches_left))
        item = self.actions.pop(0)
        if isinstance(item, Exception):
            raise item
        action, args = item
        return Decision(action, args, {"prompt_tokens": 10, "output_tokens": 2, "thinking_tokens": 0}, 1.0, 1)


class AgentLoopTests(unittest.TestCase):
    def cfg(self, **kw):
        return AgentConfig(**{"top_k": 3, "budget": 10_000, "max_searches": 2, **kw})

    def test_answer_maps_aliases_to_chunk_ids(self):
        ctrl = ScriptedController([("search", {"query": "q"}),
                                   ("answer", {"answer": "Claim [c2, c1].", "citations": ["c1", "c2"]})])
        out = run_agent("Q?", ctrl, FakeRetriever(), self.cfg())
        self.assertEqual(out["status"], "answered")
        self.assertEqual(out["citations"], ["hash1", "hash2"])
        self.assertIn("[hash2][hash1]", out["answer"])
        self.assertEqual(sum(e["event"] == "search" for e in out["events"]), 1)

    def test_citation_outside_context_abstains(self):
        ctrl = ScriptedController([("search", {"query": "q"}),
                                   ("answer", {"answer": "Claim [c9].", "citations": ["c9"]})])
        out = run_agent("Q?", ctrl, FakeRetriever(), self.cfg())
        self.assertEqual(out["status"], "abstain")
        self.assertEqual(out["events"][-1]["event"], "invalid_or_missing_citations")

    def test_search_limit_forces_final_decision(self):
        ctrl = ScriptedController([("search", {"query": "a"}), ("search", {"query": "b"}),
                                   ("abstain", {"reason": "missing"})])
        out = run_agent("Q?", ctrl, FakeRetriever(), self.cfg())
        self.assertEqual(out["status"], "abstain")
        self.assertEqual(out["searches"], 2)
        self.assertEqual(ctrl.seen[-1][2], 0)  # no searches left on the last decision

    def test_provider_error_is_reported_not_hidden(self):
        ctrl = ScriptedController([ProviderError("quota")])
        out = run_agent("Q?", ctrl, FakeRetriever(), self.cfg())
        self.assertEqual(out["status"], "error")
        self.assertEqual(out["events"][-1]["source"], "model")

    def test_budget_keeps_first_chunks_and_rank_order_interleaves(self):
        block = len(f"[c1 | DSA.pdf | page 1 | 3.1]\n{CHUNKS[0]['text']}")
        r = FakeRetriever()
        append = ContextState(budget=block * 2 + 2)
        append.add(r.search("a"), 1)
        append.add(r.search("b"), 2)
        self.assertEqual(append.ids, ["hash1", "hash2"])
        r = FakeRetriever()
        ranked = ContextState(budget=block * 2 + 2, order="rank")
        ranked.add(r.search("a"), 1)
        ranked.add(r.search("b"), 2)
        self.assertEqual(ranked.ids, ["hash1", "hash4"])


if __name__ == "__main__":
    unittest.main()
