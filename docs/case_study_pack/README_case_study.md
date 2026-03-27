# Fashion Shopping Assistant — Case Study

## What This Is

A multimodal AI shopping assistant built as a take-home prototype for an AI Engineer role. It accepts natural language queries and image uploads and returns curated product recommendations from a predefined fashion catalog, with support for multi-turn follow-up conversations and simple bundle planning.

Live demo: https://shopping-assistant-tdpvxkzwtjc5tect6wi6app.streamlit.app/

---

## Capabilities

| Feature | Description |
|---|---|
| Text search | "Find me black casual sneakers under $80 for men" |
| Image search | Upload photo → visually similar catalog items via CLIP |
| Follow-up | "Show me the casual ones" / "which is cheapest" over shown results |
| Best-one | "Which would you recommend?" → single highest-scoring item |
| Bundle | "Suggest a hoodie and sneaker combo under $150" |
| General chat | Agent identity, capabilities, scope explanation |
| Scope refusal | Out-of-domain questions refused with redirect |

---

## Architecture Summary

```
Streamlit Cloud (frontend)  ──POST /agent──▶  Railway (FastAPI backend)
                                                      │
                                    ┌─────────────────┼─────────────────────┐
                                    │                 │                     │
                               Planner          Anthropic API        Local disk
                               (rules)          (Claude Haiku)       catalog.parquet
                                    │           parse / judge /       image_embeddings.npy
                               Retriever        orchestrate          catalog_thumbnails/
                               TF-IDF / CLIP
                                    │
                               Ranker + Judge
                                    │
                               Orchestrator ──────▶ AgentResponse
```

Key design rule: deterministic retrieval and filtering; LLM used only for constraint extraction, product judgment, and response composition.

---

## Why This Design

**Retrieval before LLM.** Sending the entire catalog to an LLM is expensive, slow, and produces hallucinations. The system uses TF-IDF and CLIP to narrow ~2,500 products to a 30-item shortlist, then uses the LLM to judge that shortlist. This is cost-effective, fast, and honest — the LLM only evaluates items that actually exist.

**No fallback recommendations.** If the judge or orchestrator fails, the response is an explicit error message with zero products. A system that silently returns "best available" products when its evaluation layer is broken is a system that lies to users. Honest failure is a feature.

**Stateless backend, stateful frontend.** All conversational context (search filters, shown products, accepted pool) lives in Streamlit session state and is sent with every request. The backend is a pure function of its inputs. This makes the backend horizontally scalable and eliminates distributed state management.

**Hosted demo matters.** A working URL is the deliverable. Both services use free-tier hosting (Streamlit Community Cloud, Railway) with an API key injected as a server-side environment variable — never in source, never in the frontend.

---

## Setup

**Prerequisites:** Python 3.11+, Kaggle Fashion Product Images Dataset, Anthropic API key.

```bash
# 1. Clone and install
git clone <repo>
pip install -r requirements.txt -r requirements-backend.txt

# 2. Configure
cp .env.example .env
# Set DATASET_ROOT, ANTHROPIC_API_KEY

# 3. Preprocess catalog
python -m app.data.preprocess_dataset
python -m app.data.build_image_embeddings  # Optional; enables image search

# 4. Run backend
uvicorn app.main:app --host 0.0.0.0 --port 8000

# 5. Run frontend
API_URL=http://localhost:8000 streamlit run frontend/streamlit_app.py
```

---

## What I'd Do Next

**Dense text retrieval.** Replace TF-IDF with a sentence-transformer model (e.g., `all-MiniLM-L6-v2`) for semantic query understanding. "Cozy winter outfit" should match items without those exact words in the description.

**Persistent rate limiting.** In-memory counters reset on every server restart. A Redis-backed counter would make the demo cap reliable for a public deployment.

**Evaluation dataset.** A small labeled set of (query, expected product IDs) pairs would enable regression testing when models or prompts change. The current system has no quantitative quality baseline.

**Fashion-specific CLIP.** A model fine-tuned on fashion datasets (FashionIQ, iMaterialist) would improve attribute-level image retrieval — distinguishing striped from plain, or slim from relaxed fit — beyond what general CLIP captures.

**Streaming responses.** The orchestrator LLM call blocks until complete before the frontend renders. Streaming via Server-Sent Events would improve perceived latency on slower connections.

---

## Deployment Notes

- `ANTHROPIC_API_KEY` is a Railway environment variable. It is never in source code or sent to the frontend.
- `API_URL` is a Streamlit Cloud secret. The frontend reads it via `os.environ["API_URL"]`.
- The backend Docker image bakes CLIP model weights at build time to avoid a 600 MB download at startup.
- CPU-only PyTorch (`torch==2.3.0+cpu`) keeps the Docker image under Railway's 4 GB limit.
- Per-session limit: 100 LLM calls. Global cap: 2000 LLM calls. Both configurable via environment variables.
