# Fashion Shopping Assistant

A scoped multimodal AI shopping assistant for a predefined fashion catalog. One `POST /agent` endpoint handles general chat, text-based product search, image-based visual search, multi-turn follow-up, and simple bundle planning.

Built as a take-home prototype demonstrating a practical agent pipeline: retrieval → LLM judgment → orchestrated response, with explicit multi-turn context and a strict no-fallback-recommendation policy.

**Live demo:** https://shopping-assistant-tdpvxkzwtjc5tect6wi6app.streamlit.app/

---

## Features

| Capability | Example |
|---|---|
| General chat | "What can you help me find?" |
| Text recommendation | "Black casual sneakers under $80 for men" |
| Image search | Upload a photo → visually similar catalog items |
| Follow-up filtering | "Show me only the casual ones" / "Which is cheapest?" |
| Best-one selection | "Which one would you recommend?" |
| Bundle planning | "Suggest a hoodie and sneaker combo under $120" |
| Pagination | "Show me another" → next items from accepted pool |
| Scope refusal | Non-fashion questions refused with explanation |

---

## Tech Stack and Rationale

| Technology | Role | Why |
|---|---|---|
| **FastAPI** | Backend API | Lightweight, async, automatic OpenAPI docs; ideal for a single-endpoint prototype |
| **Streamlit** | Frontend | Fast to build, deploys free on Streamlit Community Cloud without a separate web server |
| **Claude Haiku** (Anthropic) | LLM for parsing, judging, orchestration | Fast and cheap for structured tasks; Haiku handles JSON extraction and short-form generation well without the cost of Sonnet |
| **TF-IDF** (scikit-learn) | Text retrieval | Simple, no inference latency, deterministic, sufficient for a ~2,500-product catalog |
| **CLIP ViT-B/32** (OpenAI via HuggingFace) | Image embeddings | Established baseline for visual similarity search; small enough to run on CPU at demo scale |
| **pandas + parquet** | Catalog storage | Fast to load at startup; single-file artifact, no database dependency |
| **Railway** | Backend hosting | Dockerfile-based deploy, free tier, environment variable injection for API key |
| **Streamlit Community Cloud** | Frontend hosting | Zero-config deploy from GitHub, free tier, supports secrets injection |

**Key tradeoffs:**

- **TF-IDF vs dense embeddings:** TF-IDF is keyword-based and won't match "cozy winter outfit" unless those words appear in product descriptions. A sentence-transformer would handle semantic queries better. TF-IDF was chosen for simplicity and zero inference latency at search time.
- **Flat `.npy` vs vector database:** A flat NumPy index works for ~2,500 products (cosine similarity is a single matrix multiply). FAISS or a vector DB would be needed beyond ~100k products.
- **In-memory rate limiting:** Per-session and global LLM call counters live in Python dicts and reset on server restart. Good enough for a controlled demo; production would use Redis.
- **Mock prices:** Prices are generated deterministically from a hash of product ID and category. They are structurally correct for filtering but not real USD values.

---

## Architecture

```
User input (text + optional image)
        │
        ▼
  [Rate Limiter]  ← per-session (100 calls) + global (2000 calls)
        │
        ▼
  [Planner]  ← 10-step rule cascade + LLM fallback for ambiguous short text
        │
        ├── Non-retrieval: capabilities / state / reset / out_of_scope
        │       └─▶ Response Generator (Claude Haiku or template fallback)
        │
        └── Retrieval actions
                │
                ▼
        [Query Parser]  ← Claude Haiku extracts {category, color, gender,
                           usage, price, brand}; keyword regex if no key
                │
                ▼
        [Retriever]
          TEXT:   TF-IDF cosine similarity + hard metadata filters → top 30
          IMAGE:  CLIP cosine similarity → visual reranker → top 30
                │
                ▼
        [Judge]  ← Claude Haiku evaluates shortlist
          TEXT:   strict prompt, labels each item exact_match / close_alternative / reject
          IMAGE:  multimodal prompt with query image + up to 12 product thumbnails
                │
          JudgeUnavailableError → explicit error message, zero products
                │
                ▼
        [Orchestrator]  ← Claude Haiku composes natural language response
          Grounded in shown product list; cannot hallucinate product details
                │
          LLMUnavailableError → explicit error message, zero products
                │
                ▼
        AgentResponse {message, products[], bundle_pairs[], search_state}
```

**No-fallback rule:** If the judge or orchestrator is unavailable (missing API key, network error, rate limit), the endpoint returns an explicit error message with zero products. There are no silent degraded responses. Text retrieval works without an API key, but the pipeline requires LLM judgment to produce a finished recommendation.

**Multi-turn context:** All conversational state (active filters, shown product IDs, accepted pool for pagination) lives in Streamlit session state and is round-tripped with every request. The backend is stateless.

---

## Project Structure

```
app/
  main.py                    FastAPI app factory; lifespan loads catalog + starts CLIP
                             background thread so /health responds immediately
  config.py                  Environment variable loading (.env or .env.txt)
  schemas.py                 AgentRequest / AgentResponse / SearchState Pydantic models
  api/
    routes_agent.py          POST /agent — full orchestration logic (~1,200 lines)
  services/
    planner.py               Determines action per turn (rules + optional LLM fallback)
    intent_router.py         Legacy single-turn regex classifier; patterns imported by planner.py
    query_parser.py          LLM constraint extractor + keyword regex fallback
    retriever.py             TF-IDF retriever with hard metadata filtering
    image_search.py          CLIP similarity search + visual reranker
    scorer.py                0-100 candidate scorer (used by retrieval before judging)
    judge.py                 LLM judge: exact_match / close_alternative / reject
    orchestrator.py          LLM response composer, grounded in retrieved products
    response_generator.py    General chat / clarification / scope-refusal responses
    rate_limiter.py          Per-session + global LLM call counters
  data/
    preprocess_dataset.py    Raw Kaggle dataset → catalog.parquet
    build_image_embeddings.py Catalog images → image_embeddings.npy
    build_thumbnails.py      Full-size images → 180px JPEG thumbnails
    catalog_loader.py        Load catalog + embeddings at startup
  utils/
    text_utils.py            strip_html / normalize_whitespace (used in preprocessing)
frontend/
  streamlit_app.py           Chat UI with image upload, product cards, bundle display
tests/
  test_intent_router.py      Regex pattern coverage for intent_router.py
  test_query_parser.py       Keyword fallback extraction
  test_retriever.py          TF-IDF filtering and ranking
  test_scorer.py             Candidate scorer thresholds and penalties
  test_conversational.py     Follow-up filtering logic (_filter_active_products)
  test_api_smoke.py          /agent endpoint smoke tests (no API key required)
data/
  processed/                 catalog.parquet, image_embeddings.npy, embedding_ids.npy
  catalog_thumbnails/        180px JPEG thumbnails served at /images/{id}.jpg
docs/
  case_study_pack/           Extended design and architecture documentation
```

---

## Setup

### 1. Clone and install

```bash
git clone <repo-url>
cd commerce_agent

python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# Frontend deps only (Streamlit + requests)
pip install -r requirements.txt

# Backend deps (FastAPI, torch, CLIP, etc.)
pip install -r requirements-backend.txt
```

> Streamlit Community Cloud installs `requirements.txt` only (frontend-safe).
> Railway uses `requirements-backend.txt` via Dockerfile.
> For local full-stack development, install both.

### 2. Configure environment

```bash
cp .env.example .env
```

Edit `.env`:
```
ANTHROPIC_API_KEY=sk-ant-...
DATASET_ROOT=/absolute/path/to/fashion-dataset/fashion-dataset
```

On Windows, if `.env` is saved as `.env.txt`, the config loader handles that automatically.

### 3. Preprocess the dataset

Run once locally. The outputs are committed to the repo so deployed services don't need the raw dataset.

```bash
# 1. Build catalog (~1-2 min) → data/processed/catalog.parquet
python -m app.data.preprocess_dataset

# 2. Resize images to thumbnails (~2-3 min) → data/catalog_thumbnails/
python -m app.data.build_thumbnails

# 3. Build CLIP embeddings (~3-5 min on CPU) → data/processed/image_embeddings.npy
python -m app.data.build_image_embeddings

# 4. Commit the generated artifacts
git add data/processed/catalog.parquet \
        data/catalog_thumbnails/ \
        data/processed/image_embeddings.npy \
        data/processed/embedding_ids.npy
git commit -m "Add processed catalog, thumbnails, and embeddings"
```

Steps 2–3 are only needed for image search. Text search works without them.

### 4. Run the backend

```bash
uvicorn app.main:app --reload --port 8000
```

API docs: `http://localhost:8000/docs`
Diagnostics (includes live LLM ping): `http://localhost:8000/diagnostics`

### 5. Run the frontend

```bash
API_URL=http://localhost:8000 streamlit run frontend/streamlit_app.py
```

Open `http://localhost:8501`. On Windows, set `API_URL` in your `.env` file.

---

## Usage

| Try this | What happens |
|---|---|
| "What can you do?" | Lists supported features |
| "Find me a black t-shirt for sports under $60" | Text search with constraints |
| "Show me women's casual bags" | Category + gender + usage filtering |
| Upload image + "find something similar" | CLIP visual similarity search |
| "Show me the casual ones" (after results shown) | Filters currently shown products |
| "Which is cheapest?" | Sorts shown products by price |
| "Which one would you recommend?" | Returns single best-scored item |
| "Show me a hoodie and sneaker combo" | 2-item bundle planning |
| "Show me another" | Next items from accepted pool |
| "What's the capital of France?" | Scope refusal + redirect |

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | `""` | Claude API key — server-side only, never sent to frontend |
| `DATASET_ROOT` | `data/fashion-dataset/fashion-dataset` | Path to raw Kaggle dataset (preprocessing only) |
| `CATALOG_PATH` | `data/processed/catalog.parquet` | Preprocessed catalog |
| `EMBEDDINGS_PATH` | `data/processed/image_embeddings.npy` | CLIP embeddings |
| `EMBEDDING_IDS_PATH` | `data/processed/embedding_ids.npy` | IDs parallel to embeddings |
| `MAX_REQUESTS_PER_SESSION` | `100` | LLM calls allowed per session |
| `MAX_TOTAL_REQUESTS` | `2000` | Global LLM call cap across all sessions |
| `ORCHESTRATOR_TEMPERATURE` | `0.6` | Claude temperature for response generation |
| `TRANSCRIPT_WINDOW` | `10` | Number of prior turns sent as context |
| `API_HOST` | `localhost` | Backend host (for local frontend) |
| `API_PORT` | `8000` | Backend port |

---

## Running Tests

```bash
pytest tests/ -v
```

Requires `data/processed/catalog.parquet` to exist. No API key needed.

Covers: intent routing regex patterns, query parser keyword fallback, TF-IDF retrieval filtering, candidate scorer thresholds, follow-up filter logic, and `/agent` endpoint smoke tests.

---

## Deployment

### Backend → Railway

1. Push repo to GitHub.
2. Create a new Railway service, connect the repo.
3. Railway detects `railway.toml` → builds with Dockerfile.
4. Set environment variables in Railway dashboard: `ANTHROPIC_API_KEY`, and optionally the path overrides.
5. The Dockerfile pre-downloads CLIP weights at build time so container startup is fast.

### Frontend → Streamlit Community Cloud

1. Connect repo at share.streamlit.io.
2. Set main file: `frontend/streamlit_app.py`.
3. In **Advanced Settings → Python version**: leave default (Streamlit Cloud manages this).
4. In **App settings → Secrets**, add:
   ```toml
   API_URL = "https://your-backend.railway.app"
   ```
5. Deploy. Streamlit Cloud installs `requirements.txt` (frontend-safe, 3 packages).

---

## Design Decisions

**Planner-first routing.** Every request goes through `planner.py` before any retrieval or LLM call. The planner uses a 10-step rule cascade (image attached? reset signal? capability question? bundle pattern? etc.) and falls back to a lightweight LLM call only for genuinely ambiguous short text. This makes routing fast and testable without burning tokens on easy cases.

**Retrieval before judgment.** The catalog is never sent to the LLM directly. TF-IDF or CLIP retrieves a 30-item candidate shortlist; the LLM judge evaluates only that shortlist. This keeps cost predictable and prevents hallucinated product details — the orchestrator receives an explicit product list and is instructed not to reference items not in that list.

**LLM used for judgment, not search.** Category matching, color filtering, price constraints, and gender filtering are all handled deterministically before the LLM is called. The LLM judge resolves ambiguous cases (is "charcoal grey" close enough to "black"?) that rules can't handle cleanly.

**No fallback recommendations.** A system that silently returns degraded results when its evaluation layer fails is harder to trust and debug than one that says "I can't complete this right now." If the judge or orchestrator is unavailable, the response is an explicit error message and zero products. General chat is the only path with a template fallback (it has no correctness requirement tied to catalog data).

**Stateless backend.** All conversational context — active filters, shown product IDs, accepted pool for pagination, image search mode — lives in Streamlit session state and is sent with every `POST /agent` request. The backend holds no per-user state. This makes horizontal scaling straightforward and simplifies reasoning about correctness.

---

## Limitations

- **TF-IDF retrieval is shallow.** Semantic queries without matching keywords ("something cozy for winter evenings") may return poor results.
- **Mock prices.** Prices are deterministic but not real USD values.
- **In-memory rate limiting** resets on server restart.
- **~2,500-product catalog.** The flat embedding index won't scale past ~100k products without a vector database.
- **Bundle planner is 2-item only.** No full outfit assembly, no cross-item color coordination.
- **No authentication.** Session IDs are browser-local UUIDs; a user can reset their session counter by clearing state.

---

## Future Improvements

- Dense text embeddings (sentence-transformers) for semantic query understanding
- FAISS index for larger catalog scale
- Persistent rate limiting via Redis
- Fashion-specific CLIP fine-tune for better attribute-level image retrieval
- Streaming responses via Server-Sent Events to reduce perceived latency
- Real product price data and live inventory integration
- Evaluation dataset for regression testing retrieval and judgment quality

---

## Demo Safety

- `ANTHROPIC_API_KEY` is a server-side environment variable; it is never sent to the frontend or logged.
- Per-session limit (default 100 LLM calls) and global cap (default 2000) prevent runaway cost.
- General chat has template fallbacks so the frontend remains usable if the API key is missing; all retrieval paths return explicit errors rather than degraded results.
- Do not expose this publicly without adding authentication if you lower the rate limits or increase the caps.
