# Implementation Decisions

## FastAPI + Streamlit Split (vs. Monolith)

The backend and frontend are separate deployed services rather than a single app.

**Why:** Streamlit is not a web server. Running a FastAPI server inside a Streamlit process is fragile, blocks the event loop, and doesn't survive Streamlit's re-run model. More importantly, the API key must stay server-side — a Streamlit-only app would require client-side API calls or complex workarounds to avoid exposing the key.

The two-service split has a clean contract: one `POST /agent` endpoint, one JSON schema. The frontend knows nothing about retrieval or LLM internals. This also makes the backend independently testable without a browser.

**Tradeoff:** Two deployment targets to manage. In practice this cost one extra service configuration and a `secrets.toml` — acceptable for the benefit.

## Anthropic LLM Usage (Haiku, Not Sonnet)

Claude Haiku is used for all LLM calls: query parsing, product judging, and response orchestration.

**Why:** Haiku is fast (~1-2s for short prompts), cheap, and sufficient for structured tasks like JSON extraction and short-form natural language generation. Sonnet or Opus would add cost and latency with no meaningful quality improvement for these narrow tasks.

LLM is used in four places only:
1. Query parsing (extract JSON constraints from natural language)
2. Text product judge (evaluate 30 candidates → exact/close/reject)
3. Image product judge (multimodal evaluation of product images)
4. Orchestrator (compose the final user-facing response)

Everything else — routing, filtering, ranking, rate limiting, guardrails — is deterministic code. This limits API cost and makes the system's behavior predictable.

## CLIP-Based Image Retrieval

`openai/clip-vit-base-patch32` was chosen for image similarity search.

**Why:** CLIP is the established baseline for cross-modal similarity search. It produces 512-dimensional embeddings that encode visual semantics well enough to match garment type, color, and silhouette without labeled training data. The ViT-B/32 variant is small (~350 MB weights), fast on CPU for inference at demo scale, and produces good enough results for this use case.

Product catalog embeddings are precomputed offline (`data/preprocess_dataset.py` runs `build_image_embeddings.py`) and stored as a flat `.npy` array. At query time, only the query image is encoded live; catalog similarity is a single matrix multiply.

**Tradeoff:** CLIP similarity is not perfect for fine-grained fashion attributes. A plain white t-shirt and a patterned white t-shirt may score similarly. The visual reranker applies lightweight heuristics (dominant color, pixel variance as complexity proxy) to correct the most common cases. For production, a fine-tuned fashion CLIP or DINOv2 model would perform better.

## Processed Artifact Approach (vs. Raw Dataset at Runtime)

The catalog is preprocessed offline into `data/processed/catalog.parquet` and `data/processed/image_embeddings.npy`. The runtime never reads the raw Kaggle CSV or JSON files.

**Why:** The raw dataset is ~40,000 products with per-product JSON files. Loading and processing this at startup would take 30-60 seconds, require the full Kaggle dataset to be present in the container, and repeat expensive work on every deploy. The parquet file loads in under a second. The `.npy` embeddings load in 1-2 seconds.

This pattern — precompute expensive artifacts, deploy only the outputs — is standard practice in production ML systems. It also means the Docker image does not need the raw dataset (the COPY steps in the Dockerfile copy only `data/processed/` and `data/catalog_thumbnails/`).

**Tradeoff:** Updating the catalog requires re-running the preprocessing pipeline. Acceptable for a static demo catalog; would require a proper data pipeline for a live product database.

## Result-Set Grounding for Follow-ups

Follow-up queries operate on the `active_products` list sent by the frontend — the products currently visible to the user — not the full catalog.

**Why:** This is the correct semantics. When a user says "show me the casual ones," they mean among what I can see right now, not among all catalog products. Grounding follow-ups to the shown result set also prevents the model from hallucinating products it thinks it showed.

The `SearchState.accepted_pool` additionally tracks all products that passed the judge in the prior search (up to 30), enabling "show me another" pagination without re-querying. `shown_product_ids` tracks deduplication across pages.

**Tradeoff:** If the user wants to completely change their query, the planner must correctly route to `new_search` rather than `followup_on_results`. The `_NEW_SEARCH_SIGNAL_RE` guard and planner cascade handle most cases, with LLM fallback for ambiguous short text.

## Small Planner for 2-Item Bundles

Bundle requests ("suggest a hoodie and sneaker combo") are handled by decomposing the request into two independent category searches, judging each separately, and pairing the accepted results.

**Why:** The alternative — a single search that returns mixed-category results — produces incoherent output. A query for "hoodie and sneaker" against a TF-IDF index returns whatever happens to score highest, often items from neither category. By decomposing explicitly, each category gets its own retrieval and judge pass, and pairing happens on the accepted sets.

**Scope limit:** The bundle planner handles exactly 2-category pairs. It does not build complete outfits (top + bottom + footwear + accessory). That would require a more sophisticated planner, a richer semantic schema, and substantially more complex ranking. For a demo, 2-item bundles demonstrate the concept cleanly.

## Hosted Backend + Hosted Frontend (vs. One Monolith)

The system runs as two hosted services rather than a single Streamlit app with an embedded server or a single Docker container.

**Why:** Streamlit Community Cloud is free, zero-config, and deploys directly from a GitHub repo. Railway handles the Docker backend with environment variable injection. The split keeps each service within the resource limits of its respective free tier. It also demonstrates production-appropriate architecture: API servers and web frontends are different things with different scaling and deployment concerns.

**Why not a single container:** Combining FastAPI and Streamlit in one process is technically possible but not how either framework is designed to run. It creates operational complexity (process supervision, port management, startup ordering) with no benefit for a demo. The two-service model is simpler to reason about and easier to explain.
