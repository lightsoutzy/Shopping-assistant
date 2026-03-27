# Fashion Shopping Assistant

A single AI agent for fashion e-commerce. One `POST /agent` endpoint handles general chat,
text-based product recommendations, and image-based similarity search.

Built as a take-home prototype demonstrating a clean agent pipeline with FastAPI, Streamlit,
Claude (Anthropic), TF-IDF retrieval, and CLIP image embeddings.

---

## Features

| Capability | Description |
|---|---|
| General chat | Ask what the assistant is, what it can do, what it carries |
| Text recommendation | Natural language queries with category, color, gender, budget filtering |
| Image search | Upload a photo; returns visually similar catalog products via CLIP |
| Unified endpoint | Single `POST /agent` — no separate routes per feature |
| Rate limiting | Per-session and global LLM call caps for demo safety |

---

## Architecture

```
POST /agent
  → Intent Router       rule-based (chat / recommend / image_search / clarify)
  → Query Parser        Claude Haiku extracts constraints; keyword fallback if no key
  → Retriever           TF-IDF on searchable_text + hard metadata filters
    OR Image Search     CLIP query encoding + cosine similarity vs. .npy embeddings
  → Ranker              weighted metadata scoring, top 3
  → Evaluator           rule-based constraint check; one retry on violation
  → Response Generator  Claude Haiku; template fallback if no key
```

LLM (Claude) is used only for query parsing, response generation, and general chat.
All retrieval, ranking, and guardrails are deterministic.

---

## Setup

### 1. Install dependencies

```bash
python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env and set:
#   ANTHROPIC_API_KEY=sk-ant-...
#   DATASET_ROOT=/absolute/path/to/fashion-dataset/fashion-dataset
```

> The app runs without an API key — responses fall back to templates.

### 3. Preprocess the dataset and build assets

Run these once locally. The outputs are committed to the repo so deployed services don't need the raw dataset.

```bash
# 1. Build catalog (~1-2 min)
python -m app.data.preprocess_dataset

# 2. Resize catalog images to thumbnails (~2-3 min, produces ~40 MB in data/catalog_thumbnails/)
python -m app.data.build_thumbnails

# 3. Build CLIP image embeddings (~3-5 min on CPU)
python -m app.data.build_image_embeddings

# 4. Commit the generated assets
git add data/processed/catalog.parquet data/catalog_thumbnails/ data/processed/image_embeddings.npy data/processed/embedding_ids.npy
git commit -m "Add processed catalog, thumbnails, and embeddings"
```

> Steps 2–4 are only needed for image search. The backend starts and text search works without them.

### 5. Run the backend

```bash
uvicorn app.main:app --reload --port 8000
```

API docs available at `http://localhost:8000/docs`.

### 6. Run the frontend

```bash
streamlit run frontend/streamlit_app.py
```

Open `http://localhost:8501`.

---

## Usage

| Try this | Expected behavior |
|---|---|
| "hello" | General greeting + capability summary |
| "what can you do?" | Lists supported features |
| "find me a black t-shirt for sports under $60" | Returns 3 matching products with reasons |
| "show me women's casual bags" | Filtered bag results |
| Upload image + "find something similar" | CLIP-based visual similarity results |

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | `""` | Claude API key (server-side only) |
| `DATASET_ROOT` | see `.env.example` | Path to Kaggle dataset folder |
| `CATALOG_PATH` | `data/processed/catalog.parquet` | Preprocessed catalog |
| `EMBEDDINGS_PATH` | `data/processed/image_embeddings.npy` | CLIP embeddings |
| `EMBEDDING_IDS_PATH` | `data/processed/embedding_ids.npy` | IDs parallel to embeddings |
| `MAX_REQUESTS_PER_SESSION` | `20` | LLM calls per session |
| `MAX_TOTAL_REQUESTS` | `500` | Global LLM call cap |
| `API_HOST` | `localhost` | Backend host (used by Streamlit) |
| `API_PORT` | `8000` | Backend port |

---

## Running Tests

```bash
pytest tests/ -v
```

Covers: intent routing, query parser fallback, retriever filtering, and `/agent` endpoint smoke tests.
Requires `catalog.parquet` to exist. No API key needed.

---

## Project Structure

```
app/
  main.py                  FastAPI app + startup (catalog load, TF-IDF index, CLIP)
  config.py                Settings from .env
  schemas.py               AgentRequest / AgentResponse Pydantic models
  api/routes_agent.py      POST /agent orchestration
  services/
    intent_router.py       Rule-based intent classifier
    query_parser.py        LLM constraint extractor + keyword fallback
    retriever.py           TF-IDF retriever with metadata filtering
    image_search.py        CLIP similarity search
    ranker.py              Weighted metadata scorer
    response_generator.py  Claude response generation + template fallback
    evaluator.py           Rule-based guardrail check
    rate_limiter.py        Per-session + global LLM call limits
  data/
    preprocess_dataset.py  Raw dataset → catalog.parquet
    build_image_embeddings.py  Catalog images → .npy embeddings
    catalog_loader.py      Load catalog at startup
frontend/
  streamlit_app.py         Chat UI with image upload and product cards
tests/
  test_intent_router.py
  test_query_parser.py
  test_retriever.py
  test_api_smoke.py
```

---

## Design Tradeoffs

| Decision | Tradeoff |
|---|---|
| TF-IDF over dense embeddings | Simpler, no inference latency; less semantic for ambiguous queries |
| Flat `.npy` over vector DB | No infra; fine for ~2k products; won't scale past ~100k |
| In-memory rate limiting | Resets on restart; sufficient for a demo |
| Mock prices | Reproducible and structurally correct; not real USD prices |
| LLM for query parsing | Handles varied phrasing well; one extra LLM call per recommendation |

---

## Future Improvements

- Dense text embeddings for better semantic recall
- Multi-turn conversation context
- Real product prices
- FAISS index for larger catalogs
- Auth + persistent rate limiting for public deployment
- Product comparison responses

---

## Demo Safety

- API key is server-side only; never sent to the frontend
- Per-session limit (default 20) and global cap (default 500) on LLM calls
- Graceful fallback if key is missing — app still runs with template responses
- Do not expose publicly without adding authentication
