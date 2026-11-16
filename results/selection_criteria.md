# Configuration selection criteria (fixed 2026-11-16, before running improvement configs and before any test run)

Selection uses **dev only**. Criteria are applied in order; a later criterion only breaks ties of the earlier ones.

1. **No provider/vector-store errors** on dev (status `error` = 0).
2. **Grounded answer mean (/2)** on the 15 answerable dev questions, manual labels.
3. **Abstention**: unanswerable dev questions abstained (must be 2/2), then fewest false abstentions on answerable questions.
4. **Citation validity**: supported citations / total citations (manual labels).
5. **Context Evidence Rate** of the final agent context (gold evidence verbatim in a context chunk).
6. **Cost/latency**: fewer total tokens (prompt + output + thinking), then lower median latency per question.

Candidate configurations (agent = gemini-3.8-flash, temperature 0, max 3 searches):

| Name | Retriever | top-k | Context budget (chars) | Context order | Role |
|---|---|---|---|---|---|
| dense_k3_b2500_s3 | Dense pgvector (Day 6) | 3 | 2500 | append | Baseline |
| bm25_k3_b2500_s3 | BM25 | 3 | 2500 | append | Method 2 |
| hybrid_k3_b2500_s3 | Hybrid RRF (dense + BM25) | 3 | 2500 | append | Method 3 |
| hybrid_k3_b2500_s3_rank | Hybrid RRF | 3 | 2500 | rank | Improvement 1 |
| hybrid_nb_k3_b5000_s3 | Hybrid RRF + neighbour chunks (+-1) | 3 (+neighbours) | 5000 | append | Improvement 2 |

After selection, the chosen configuration and the baseline are run once on test with no further changes.

## Decision (dev results, before any test run)

| Config | Errors | Grounded 0/1/2 (mean) | Abstain / false abstain | Citations | Context evidence | Tokens in/out+think | Median s |
|---|---|---|---|---|---|---|---|
| dense_k3_b2500_s3 | 0 | 3/1/11 (1.53) | 2/2 / 3/15 | 13/14 | 10/15 | 41.3k / 9.2k | 6.65 |
| bm25_k3_b2500_s3 | 0 | 1/1/13 (1.80) | 2/2 / 1/15 | 17/18 | 12/15 | 35.9k / 7.0k | 7.33 |
| hybrid_k3_b2500_s3 | 0 | 0/1/14 (1.93) | 2/2 / 0/15 | 20/20 | 14/15 | 35.0k / 7.8k | 6.96 |
| hybrid_k3_b2500_s3_rank | 0 | 0/1/14 (1.93) | 2/2 / 0/15 | 19/19 | 13/15 | 35.6k / 7.0k | 5.94 |
| **hybrid_nb_k3_b5000_s3** | 0 | **0/0/15 (2.00)** | 2/2 / 0/15 | 24/24 | **15/15** | 49.5k / 8.2k | 6.40 |

**Selected: `hybrid_nb_k3_b5000_s3`** (criterion 2: highest grounded mean; ties on criteria 3-4).
Trade-off accepted: about +42% prompt tokens vs hybrid_k3_b2500_s3. Baseline for test: `dense_k3_b2500_s3`.
