# Architecture

## System Overview

A single unified shopping assistant. One `POST /agent` endpoint handles all user interactions.
The system routes internally to specialized handlers but presents a single conversational interface.

---

## Request Pipeline

```
User Request
  ├── text: str  (required)
  └── image: str (optional, base64-encoded)
         │
         ▼
  ┌─────────────────┐
  │  Intent Router  │  ← rule-based; LLM fallback only for ambiguous cases
  └────────┬────────┘
           │
     ┌─────┴──────────────────────────────┐
     │                                    │
  general_chat               recommend_products / image_search / clarify_request
     │                                    │
     ▼                                    ▼
  LLM response            ┌──────────────────────────────┐
  (scoped to              │         Query Parser          │  ← LLM-assisted
   assistant identity     │  extracts: category, color,   │
   and capabilities)      │  gender, usage, season, price │
                          └──────────────┬───────────────┘
                                         │
                          ┌──────────────┴──────────────┐
                          │                             │
                     text query                   image query
                          │                             │
                          ▼                             ▼
                   ┌─────────────┐            ┌──────────────────┐
                   │  Retriever  │            │   Image Search   │
                   │  (TF-IDF)   │            │ (CLIP + cosine)  │
                   └──────┬──────┘            └────────┬─────────┘
                          │                            │
                          └────────────┬───────────────┘
                                       ▼
                              ┌─────────────────┐
                              │     Ranker      │  ← metadata scoring, no LLM
                              │  (top 3 output) │
                              └────────┬────────┘
                                       ▼
                          ┌────────────────────────┐
                          │   Response Generator   │  ← LLM
                          │ builds natural language │
                          │ reply with product info │
                          └────────────┬───────────┘
                                       ▼
                              ┌─────────────────┐
                              │    Evaluator    │  ← rule-based constraint check
                              │ (guardrail pass)│    retry once on violation
                              └────────┬────────┘
                                       ▼
                               Final JSON Response
```

---

## Component Responsibilities

### Intent Router (`services/intent_router.py`)
- **Input:** raw text, whether image is present
- **Output:** one of `general_chat`, `recommend_products`, `image_search`, `clarify_request`
- **Method:** keyword/pattern rules first (covers ~90% of cases); small LLM call only if confidence is low
- **Decision:** avoids unnecessary LLM latency for common requests

### Query Parser (`services/query_parser.py`)
- **Input:** user text query
- **Output:** structured constraint dict
  ```json
  {
    "category": "t-shirt",
    "color": "black",
    "gender": null,
    "usage": "sports",
    "season": null,
    "max_price": null,
    "free_text": "comfortable t-shirt for sports"
  }
  ```
- **Method:** LLM (Claude) with a short structured prompt; reliable for flexible natural language
- **Fallback:** empty constraints if LLM fails; retrieval still runs on free_text

### Retriever (`services/retriever.py`)
- **Input:** constraint dict + free_text
- **Output:** top 20–50 candidate products
- **Method:**
  1. Hard filter on extracted constraints (category, gender, color, price) against catalog DataFrame
  2. TF-IDF similarity on `searchable_text` for semantic ranking within filtered set
  3. If filter yields < 10 results, relax constraints and retry
- **No LLM**

### Image Search (`services/image_search.py`)
- **Input:** PIL Image
- **Output:** top 20–50 candidate products by visual similarity
- **Method:** encode query image with CLIP → cosine similarity vs. precomputed `.npy` catalog embeddings
- **Storage:** `data/processed/image_embeddings.npy` (shape: [N, 512]) + `data/processed/embedding_ids.npy`
- **No LLM**

### Ranker (`services/ranker.py`)
- **Input:** candidate products list + parsed constraints
- **Output:** top 3 products with scores
- **Method:** additive weighted score per product:
  - category exact match: +3
  - color match: +2
  - usage match: +2
  - gender match: +1
  - season match: +1
  - within price range: +2
  - retrieval/similarity score: normalized 0–3
- **No LLM**

### Response Generator (`services/response_generator.py`)
- **Input:** top 3 products + original user query + intent
- **Output:** natural language reply string
- **Method:** LLM (Claude) with structured system prompt
  - For recommendations: describes each product, why it fits the query
  - For general chat: scoped to assistant identity and capabilities
  - Strict instruction: only reference facts present in the provided product data

### Evaluator (`services/evaluator.py`)
- **Input:** recommended product IDs + parsed constraints
- **Output:** `{passed: bool, violations: list[str]}`
- **Method:** rule-based checks only (no LLM):
  - All returned products are in the catalog
  - Category constraint is not obviously violated
  - Price constraint is not violated
- **On failure:** pipeline retries once with a tighter retrieval pass; if still failing, returns safe fallback message

### Rate Limiter (`services/rate_limiter.py`)
- Per-session: max LLM calls configurable via `MAX_REQUESTS_PER_SESSION` (default: 20)
- Global: max total LLM calls configurable via `MAX_TOTAL_REQUESTS` (default: 500)
- Implementation: in-memory Python dict keyed by session_id; global counter
- On limit hit: returns friendly "demo limit reached" message without calling LLM

---

## LLM Usage Summary

| Pipeline Step      | LLM Used?         | Why                                      |
|--------------------|-------------------|------------------------------------------|
| Intent routing     | Fallback only      | Rules handle most cases; saves latency  |
| Query parsing      | Yes (Claude)       | Flexible NL constraint extraction       |
| Text retrieval     | No                 | TF-IDF is sufficient for 2k products    |
| Image similarity   | No                 | CLIP handles this natively              |
| Ranking            | No                 | Deterministic scoring is more reliable  |
| Response generation| Yes (Claude)       | Natural language output quality         |
| General chat       | Yes (Claude)       | Requires NL understanding               |
| Guardrail check    | No                 | Rules are faster and more predictable   |

---

## Technology Choices

| Component          | Choice                        | Reason                                                   |
|--------------------|-------------------------------|----------------------------------------------------------|
| Backend            | FastAPI                       | Lightweight, async, clean API docs via OpenAPI           |
| Frontend           | Streamlit                     | Fastest working UI, no JS, good for demos               |
| LLM                | Anthropic Claude (server-side)| API key never leaves backend                            |
| Text retrieval     | TF-IDF (scikit-learn)         | No infra, fast enough for 2k products, easy to explain  |
| Image embeddings   | CLIP clip-vit-base-patch32    | Strong visual similarity, ~600MB, CPU-viable            |
| Embedding storage  | Flat .npy + numpy cosine      | No vector DB infra needed at this scale                 |
| Catalog storage    | Parquet (pandas)              | Single file, no DB server, fast read                    |
| Rate limiting      | In-memory Python              | Sufficient for demo; no Redis needed                    |
| Pricing            | Deterministic hash-based mock | Reproducible, structurally correct, no real data needed |

---

## Key Design Constraints

- No API key ever sent to frontend
- Catalog is fixed and precomputed (not dynamic)
- Image embeddings are precomputed offline; no CLIP inference at query time for text search
- LLM calls are the only external network dependency at runtime
- All retrieval and ranking is deterministic given the same catalog
