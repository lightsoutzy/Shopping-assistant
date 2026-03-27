# Architecture

## Deployment Topology

```
User Browser
    │
    ▼
Streamlit Community Cloud
  frontend/streamlit_app.py
  (Python 3.14, streamlit==1.40.0)
    │
    │  POST /agent  (HTTPS)
    │  {text, image_b64, session_id, chat_history,
    │   active_products, search_state}
    ▼
Railway (Docker container, python:3.11-slim)
  FastAPI backend
  app/main.py
    │
    ├── Anthropic API (claude-haiku-4-5)  [external HTTPS]
    └── Local disk: catalog.parquet, *.npy, catalog_thumbnails/
```

Secrets flow:
- `ANTHROPIC_API_KEY`: Railway environment variable → loaded by `app/config.py` → never sent to frontend
- `API_URL`: Streamlit Cloud secret → read by `os.environ["API_URL"]` in frontend → used as HTTP base URL

## Full Agent Pipeline

```
User input (text + optional image_b64)
        │
        ▼
[1] Rate Limiter
    Check per-session (100 calls) and global (2000 calls) LLM caps.
    Reject with error message if exceeded.
        │
        ▼
[2] Planner  (app/services/planner.py)
    10-step rule cascade + optional LLM fallback for ambiguous short text.
    Outputs one action:
      new_search | refine_search | followup_on_results | image_search |
      bundle_search | next_result | compare_results | ask_capabilities |
      ask_current_state | explain_match_quality | reset_context | out_of_scope
        │
        ├─── Non-retrieval actions (capabilities, state, reset, out_of_scope)
        │    └──▶ Response Generator / direct response → [9]
        │
        └─── Retrieval actions
                 │
                 ▼
        [3] Query Parser  (app/services/query_parser.py)
            LLM (Haiku) extracts structured constraints:
            {category, color, gender, usage, season,
             max_price, min_price, brand, plain_only}
            Falls back to keyword regex if LLM unavailable.
                 │
                 ▼
        [4] Retriever  (app/services/retriever.py  OR  image_search.py)
            TEXT PATH:
              Hard metadata filter on catalog.parquet
              → TF-IDF cosine similarity (unigram+bigram)
              → top 30 candidates
            IMAGE PATH:
              CLIP encode query image
              → cosine similarity against pre-built embedding index
              → top 30 candidates
              → visual reranker (color family, complexity heuristics)
                 │
                 ▼
        [5] Ranker  (app/services/ranker.py)
            Weighted additive score:
            TF-IDF(3) + category(3) + color(2) + usage(2) + gender(1) + season(1) + price(2)
            → sorted shortlist
                 │
                 ▼
        [6] Judge  (app/services/judge.py)
            LLM (Haiku) evaluates each candidate:
              TEXT judge: text-only, strict constraints
              IMAGE judge: multimodal (query image + up to 12 product images)
            Verdicts: exact_match | close_alternative | reject
            Raises JudgeUnavailableError if LLM unavailable (no fallback).
                 │
                 ▼
        [7] Evaluator  (app/services/evaluator.py)
            Rule-based guardrail: confirm product IDs exist, price within bounds.
            Soft check — logs violations, does not block response.
                 │
                 ▼
        [8] Orchestrator  (app/services/orchestrator.py)
            LLM (Haiku) composes natural language response.
            Receives: shown products, exact/close counts, constraints, action label.
            Strict prompt prevents hallucination; wording rules enforce directness.
            Raises LLMUnavailableError if LLM unavailable (no fallback).
                 │
                 ▼
        [9] AgentResponse
            {intent, message, products[], bundle_pairs[], context_action, search_state}
                 │
                 ▼
        Frontend renders product cards + chat message
```

## Follow-up Memory Path

```
Frontend sends with every request:
  active_products: list[dict]   ← products currently shown in UI
  search_state: SearchState     ← {category, color, filters, accepted_pool, ...}

Planner detects followup_on_results action.
_handle_followup() in routes_agent.py:
  - Applies hard filters to active_products (color, usage, brand keyword)
  - If "best one" query: sorts by match_score, picks top 1
  - Passes filtered set to Orchestrator with filter_context
  - Returns search_state=ss (preserves context for next turn)
```

SearchState persists in frontend `st.session_state` and is round-tripped with every request. The backend is stateless — all context lives in the frontend and is sent with each call.

## Image Search Path

```
User uploads image (JPEG or PNG) via sidebar uploader.
Frontend:
  - Deduplicates via SHA-256 hash of raw bytes
  - Re-encodes as JPEG (handles RGBA PNG for API compatibility)
  - Base64-encodes → image_b64 field in request

Backend:
  - Decodes base64 → PIL Image → re-encode as JPEG (normalize media type)
  - CLIP processor tokenizes → model.encode() → 512-dim embedding
  - Cosine similarity against pre-built index (image_embeddings.npy)
  - Visual reranker adjusts scores by dominant color and complexity
  - Judge receives query image + up to 12 product thumbnails (multimodal)
  - Orchestrator composes response noting dominant color if detected
```

CLIP model (`openai/clip-vit-base-patch32`) weights are baked into the Docker image layer at build time. In-memory load happens in a background thread after startup so `/health` is immediately available.

## Simple Bundle Planner

```
User: "suggest a hoodie and sneaker combo under $120"

Planner → bundle_search

_extract_bundle_categories():
  - Regex match against _BUNDLE_CAT_MAP keywords
  - Identifies two canonical categories (e.g., "hoodie", "sneaker")

_run_bundle_search():
  - Run full text search pipeline independently for each category
  - Judge each category's shortlist
  - Collect accepted items for category A and category B
  - Build all valid pairs where pair.total_price <= budget (if specified)
  - Rank pairs by combined match_score
  - Return top 3 pairs as bundle_pairs field in AgentResponse

Frontend renders each bundle as a grouped card row with combined total price.
```

If either category yields no accepted items, the response says so explicitly. No fake bundles.

## Secrets and Config Handling

All sensitive values are environment variables, never in source:

| Secret | Location | How accessed |
|---|---|---|
| `ANTHROPIC_API_KEY` | Railway env vars | `app/config.py` → `os.getenv` |
| `API_URL` | Streamlit Cloud secrets (TOML) | `os.environ["API_URL"]` in frontend |

`app/config.py` attempts to load `.env` (Unix) then `.env.txt` (Windows workaround) via `python-dotenv`. In deployed Railway, neither file is present — environment variables are injected directly by the platform.

`.gitignore` excludes `.env`, `.env.txt`, `.env.local`, and `.streamlit/secrets.toml`. The file `.streamlit/secrets.toml.example` is committed as documentation.
