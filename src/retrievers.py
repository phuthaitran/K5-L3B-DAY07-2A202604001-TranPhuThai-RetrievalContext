"""Retrievers for Day 7.

1. DenseRetriever   - Day 6 PostgreSQL + pgvector, query embedded with the same
                       sentence-transformers/all-MiniLM-L6-v2 model (384 dims, normalized),
                       exact cosine search (no ANN index).
2. BM25Retriever    - pure-Python BM25Okapi over the exported Day 6 chunks.
3. HybridRetriever  - weighted Reciprocal Rank Fusion of Dense + BM25 candidates.
4. TfidfRetriever   - wrapper around the starter TF-IDF baseline (reference only).

Every retriever returns chunks with: id, text, source_id, source, page, metadata,
score, rank, method. Vector store errors raise RetrieverError; there is no silent
fallback to another retriever.
"""
import math
import os
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

# The embedding model is cached locally from Day 6; avoid network calls to the HF hub.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384


class RetrieverError(RuntimeError):
    """Raised when a retrieval backend (e.g. the vector store) fails."""


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", unicodedata.normalize("NFC", text).casefold())


def _matches(chunk: dict[str, Any], filters: dict[str, Any] | None) -> bool:
    if not filters:
        return True
    meta = chunk.get("metadata", {})
    return all(str(meta.get(k)) == str(v) for k, v in filters.items())


def _result(chunk: dict[str, Any], score: float, rank: int, method: str) -> dict[str, Any]:
    return {
        "id": chunk["id"],
        "source_id": chunk["source_id"],
        "source": chunk["source"],
        "page": chunk["page"],
        "text": chunk["text"],
        "metadata": chunk.get("metadata", {}),
        "score": float(score),
        "rank": rank,
        "method": method,
    }


class DenseRetriever:
    """Dense retrieval on the Day 6 pgvector table `dsa_chunks` (Strategy B)."""

    name = "dense"

    def __init__(
        self,
        database_url: str | None = None,
        model_name: str = EMBEDDING_MODEL,
        strategy: str = "B",
    ) -> None:
        import psycopg
        from sentence_transformers import SentenceTransformer

        self.database_url = database_url or os.environ.get("DATABASE_URL")
        if not self.database_url:
            raise RetrieverError("DATABASE_URL is not set (see .env.example)")
        self.strategy = strategy
        self._psycopg = psycopg
        self.model = SentenceTransformer(model_name)
        get_dim = getattr(self.model, "get_embedding_dimension", None) or self.model.get_sentence_embedding_dimension
        dim = get_dim()
        if dim != EMBEDDING_DIM:
            raise RetrieverError(f"query embedding dim {dim} != index dim {EMBEDDING_DIM}")
        self._conn = None

    def _connection(self):
        if self._conn is None or self._conn.closed:
            self._conn = self._psycopg.connect(self.database_url, autocommit=True)
        return self._conn

    def embed_query(self, query: str) -> list[float]:
        return self.model.encode(query, normalize_embeddings=True).tolist()

    def search(
        self,
        query: str,
        *,
        filters: dict[str, Any] | None = None,
        top_k: int = 3,
    ) -> list[dict[str, Any]]:
        vec = "[" + ",".join(f"{x:.8f}" for x in self.embed_query(query)) + "]"
        where, params = ["strategy = %s"], [self.strategy]
        for k, v in (filters or {}).items():
            where.append("metadata ->> %s = %s")
            params.extend([k, str(v)])
        sql = f"""
            SELECT chunk_id, document_id, source, page, content, metadata,
                   1 - (embedding <=> %s::vector) AS score
            FROM dsa_chunks
            WHERE {' AND '.join(where)}
            ORDER BY embedding <=> %s::vector
            LIMIT %s
        """
        try:
            rows = self._connection().execute(sql, [vec, *params, vec, top_k]).fetchall()
        except self._psycopg.Error as exc:
            self._conn = None
            raise RetrieverError(f"pgvector query failed: {exc}") from exc
        return [
            _result(
                {"id": r[0], "source_id": r[1], "source": r[2], "page": r[3], "text": r[4],
                 "metadata": r[5] if isinstance(r[5], dict) else {}},
                r[6], rank, self.name,
            )
            for rank, r in enumerate(rows, start=1)
        ]


class BM25Retriever:
    """BM25Okapi over the exported chunks (same corpus as the pgvector index)."""

    name = "bm25"

    def __init__(self, chunks: list[dict[str, Any]], k1: float = 1.5, b: float = 0.75) -> None:
        self.chunks = chunks
        self.k1, self.b = k1, b
        self.term_freqs = [Counter(tokenize(c["text"])) for c in chunks]
        self.doc_lens = [sum(tf.values()) for tf in self.term_freqs]
        self.avgdl = sum(self.doc_lens) / max(1, len(chunks))
        df: Counter[str] = Counter()
        for tf in self.term_freqs:
            df.update(tf.keys())
        n = len(chunks)
        self.idf = {t: math.log((n - f + 0.5) / (f + 0.5) + 1.0) for t, f in df.items()}

    def search(
        self,
        query: str,
        *,
        filters: dict[str, Any] | None = None,
        top_k: int = 3,
    ) -> list[dict[str, Any]]:
        q_terms = [t for t in tokenize(query) if t in self.idf]
        scored = []
        for idx, chunk in enumerate(self.chunks):
            if not _matches(chunk, filters):
                continue
            tf, dl = self.term_freqs[idx], self.doc_lens[idx]
            score = 0.0
            for term in q_terms:
                f = tf.get(term, 0)
                if f:
                    score += self.idf[term] * f * (self.k1 + 1) / (
                        f + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
            if score > 0:
                scored.append((score, chunk))
        scored.sort(key=lambda x: (-x[0], x[1]["id"]))
        return [_result(c, s, rank, self.name) for rank, (s, c) in enumerate(scored[:top_k], start=1)]


class HybridRetriever:
    """Weighted RRF: score(d) = sum_r w_r / (rrf_k + rank_r(d)) over dense and BM25 candidates."""

    name = "hybrid"

    def __init__(
        self,
        dense_retriever: DenseRetriever,
        lexical_retriever: BM25Retriever,
        rrf_k: int = 60,
        dense_weight: float = 0.5,
        lexical_weight: float = 0.5,
        candidate_k: int = 10,
    ) -> None:
        self.dense = dense_retriever
        self.lexical = lexical_retriever
        self.rrf_k = rrf_k
        self.weights = {"dense": dense_weight, "bm25": lexical_weight}
        self.candidate_k = candidate_k

    def search(
        self,
        query: str,
        *,
        filters: dict[str, Any] | None = None,
        top_k: int = 3,
    ) -> list[dict[str, Any]]:
        cand_k = max(top_k, self.candidate_k)
        runs = {
            "dense": self.dense.search(query, filters=filters, top_k=cand_k),
            "bm25": self.lexical.search(query, filters=filters, top_k=cand_k),
        }
        chunks: dict[str, dict[str, Any]] = {}
        fused: dict[str, float] = {}
        source_ranks: dict[str, dict[str, int]] = {}
        for run_name, results in runs.items():
            for item in results:
                cid = item["id"]
                chunks.setdefault(cid, item)
                fused[cid] = fused.get(cid, 0.0) + self.weights[run_name] / (self.rrf_k + item["rank"])
                source_ranks.setdefault(cid, {})[run_name] = item["rank"]
        ordered = sorted(fused, key=lambda cid: (-fused[cid], cid))[:top_k]
        out = []
        for rank, cid in enumerate(ordered, start=1):
            item = _result(chunks[cid], fused[cid], rank, self.name)
            item["source_ranks"] = source_ranks[cid]
            out.append(item)
        return out


class TfidfRetriever:
    """Starter TF-IDF baseline exposed through the common retriever interface."""

    name = "tfidf"

    def __init__(self, chunks: list[dict[str, Any]]) -> None:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "starter"))
        from baseline import TfidfIndex

        self.chunks = chunks
        self.index = TfidfIndex(chunks)

    def search(
        self,
        query: str,
        *,
        filters: dict[str, Any] | None = None,
        top_k: int = 3,
    ) -> list[dict[str, Any]]:
        results = [r for r in self.index.search(query, top_k=len(self.chunks)) if _matches(r, filters)]
        return [_result(r, r["score"], rank, self.name) for rank, r in enumerate(results[:top_k], start=1)]


class NeighborExpander:
    """Wrap a retriever and append the adjacent chunks (same source, document order) of each hit.

    Day 6 chunking can split one paragraph across two chunks, so the retrieved chunk may hold
    the setup while its neighbour holds the answer. Output = the base top_k hits in their
    original order, followed by neighbours grouped by parent rank (prev before next).
    Neighbours inherit the parent's score/rank and carry `neighbor_of`. Ranking metrics are
    still computed on the base hits only.
    """

    def __init__(self, base, chunks: list[dict[str, Any]], window: int = 1) -> None:
        self.base = base
        self.window = window
        self.name = f"{base.name}+nb{window}"
        self.order = sorted(chunks, key=lambda c: (c["source_id"], c["page"], c["metadata"].get("char_start", 0)))
        self.pos = {c["id"]: i for i, c in enumerate(self.order)}

    def search(
        self,
        query: str,
        *,
        filters: dict[str, Any] | None = None,
        top_k: int = 3,
    ) -> list[dict[str, Any]]:
        hits = self.base.search(query, filters=filters, top_k=top_k)
        out, seen = list(hits), {h["id"] for h in hits}
        for h in hits:
            i = self.pos.get(h["id"])
            if i is None:
                continue
            for j in [*range(i - self.window, i), *range(i + 1, i + self.window + 1)]:
                if not 0 <= j < len(self.order):
                    continue
                n = self.order[j]
                if n["id"] in seen or n["source_id"] != h["source_id"] or not _matches(n, filters):
                    continue
                seen.add(n["id"])
                item = _result(n, h["score"], h["rank"], self.name)
                item["neighbor_of"] = h["id"]
                out.append(item)
        return out


def build_retriever(name: str, chunks: list[dict[str, Any]], dense: DenseRetriever | None = None, **kwargs):
    """Factory used by the benchmark and agent scripts. Suffix `_nb` adds neighbour expansion."""
    if name.endswith("_nb"):
        return NeighborExpander(build_retriever(name[:-3], chunks, dense=dense, **kwargs), chunks, window=1)
    if name == "bm25":
        return BM25Retriever(chunks)
    if name == "tfidf":
        return TfidfRetriever(chunks)
    dense = dense or DenseRetriever()
    if name == "dense":
        return dense
    if name == "hybrid":
        return HybridRetriever(dense, BM25Retriever(chunks), **kwargs)
    raise ValueError(f"unknown retriever: {name}")
