# Submission Blurbs

---

## 120-Word Summary

This project is a multimodal AI shopping assistant built on a predefined fashion catalog. Users interact through a chat interface to find products via natural language ("black casual jacket under $100 for men"), upload a reference image to find visually similar items, and refine results through follow-up questions without starting a new search. Internally, TF-IDF retrieval and CLIP image embeddings narrow the catalog to a shortlist, which Claude Haiku judges for relevance and then composes into a response. The backend is a FastAPI service on Railway; the frontend is Streamlit on Streamlit Community Cloud. The API key is server-side only. The system enforces a strict no-fallback rule: if the LLM pipeline fails, it returns an explicit error — never a silent degraded response.

---

## 250-Word Summary

This project is a scoped, multimodal fashion shopping assistant designed as an AI Engineer take-home prototype. It demonstrates a practical end-to-end agent pipeline: retrieval, ranking, LLM judgment, multi-turn context management, and multimodal input — deployed and live on public hosting.

The user interacts through a conversational Streamlit UI. Natural language queries are parsed for structured constraints (category, color, gender, usage, price range, brand), which drive TF-IDF retrieval over a preprocessed catalog of ~2,500 products. For image queries, CLIP embeddings encode visual similarity, with color and complexity heuristics applied as a reranking step. In both cases, a shortlist of 30 candidates is evaluated by Claude Haiku, which labels each item as an exact match, close alternative, or reject. Claude then composes the final response grounded in the accepted product set.

The system supports multi-turn follow-up over shown results ("show me the casual ones," "which is cheapest"), best-one selection, and 2-item bundle planning. Context (active filters, shown product IDs, accepted pool) persists across turns via Streamlit session state, round-tripped with every request to a stateless backend.

Key design decisions: no fallback recommendations (LLM failure returns an explicit error, not a degraded response); server-side API key only; CPU-only PyTorch to fit Railway's 4 GB image limit; background CLIP loading to unblock the healthcheck; and a strict orchestrator prompt that prevents hallucination by grounding all product references in the retrieved set.

---

## 500-Word Case Study Summary

### Building a Multimodal Fashion Shopping Assistant

The goal was to build a single AI agent for a fashion e-commerce scenario: one interface, one API endpoint, one coherent user experience — supporting text search, image-based similarity search, and multi-turn follow-up conversations over results.

**Architecture**

The system runs as two hosted services: a FastAPI backend on Railway and a Streamlit frontend on Streamlit Community Cloud. A single `POST /agent` endpoint accepts text, an optional base64-encoded image, conversation history, the currently shown products, and the accumulated search state. It returns a natural language message, up to five product cards, or bundle pairs — plus an updated search state for the next turn.

The pipeline inside the backend follows a deliberate separation: deterministic components (a rule-based planner, TF-IDF retrieval, CLIP similarity search, weighted metadata ranker, rule-based evaluator) handle everything that does not require semantic judgment. The LLM (Claude Haiku) is called for exactly three tasks: extracting structured constraints from natural language, judging a 30-item candidate shortlist for relevance, and composing the final response grounded in the accepted product set.

**Multi-Turn Context**

Conversational state — active filters, shown product IDs, the full accepted pool for pagination, and image search context — is maintained in Streamlit session state and sent with every request. The backend is stateless. Follow-up queries operate on the `active_products` list from the prior turn: "show me the casual ones" filters the three to five items currently visible, not the full catalog.

**Image Search**

CLIP (`openai/clip-vit-base-patch32`) encodes the query image to a 512-dimensional embedding, which is compared against precomputed catalog embeddings via cosine similarity. A visual reranker adjusts scores by dominant color family match and pixel variance (as a proxy for plain vs. patterned). The image judge sends the query image and up to 12 product thumbnail images to Claude Haiku in a multimodal message for final evaluation.

**Deployment Challenges**

Several practical problems arose during deployment. PyTorch's default Linux wheel includes CUDA support (~3.5 GB), which exceeded Railway's 4 GB image size limit; the fix was the CPU-only wheel. Railway's default Nixpacks builder lacks pip in the Nix Python environment; the fix was a plain Dockerfile. CLIP model loading blocked the FastAPI lifespan hook before `/health` could respond; the fix was baking weights into the image layer and loading in a background thread. Streamlit Community Cloud's Python 3.14 runtime conflicted with `streamlit==1.33.0`'s Pillow dependency; the fix was upgrading to `streamlit==1.40.0`.

**Design Decisions Worth Noting**

The most deliberate decision was the no-fallback rule. When the LLM judge or orchestrator is unavailable, the response is an explicit error message and zero products. A system that silently returns its "best available" results when its evaluation layer is broken obscures failure. Honest communication about system state is a better user experience than a plausible-looking but unreliable one.

The second notable decision was the scope boundary. The agent answers fashion questions and refuses everything else with a polite redirect. Scope discipline makes the system testable, explainable, and honest about what it can actually do — which is more valuable in a portfolio context than an open-ended assistant that claims to do everything.

---

## Resume / Project Bullets

- Built a multimodal fashion shopping assistant (FastAPI + Streamlit) supporting text recommendation, CLIP image search, multi-turn follow-up, and 2-item bundle planning over a 2,500-product catalog
- Designed a hybrid retrieval pipeline: TF-IDF + CLIP for candidate generation, Claude Haiku for semantic judgment and response orchestration, with deterministic ranker and evaluator separating retrieval from LLM logic
- Deployed to Railway (Docker) and Streamlit Community Cloud; resolved Railway 4 GB image size constraint by pinning CPU-only PyTorch, fixed CLIP startup blocking healthcheck with background thread loading
- Implemented stateless backend with full context round-tripped from frontend (active filters, shown product IDs, accepted pool), enabling multi-turn follow-up without server-side session management
- Enforced strict no-fallback policy: LLM pipeline failure returns explicit error and zero products; orchestrator prompt grounds all product references in retrieved catalog records, preventing hallucination

---

## Interview Talk Tracks

### Talk Track 1: "Walk me through the architecture"

"The system is a single `POST /agent` endpoint that handles everything. When a request comes in, a rule-based planner classifies the intent — new search, follow-up, image query, bundle request, general chat. For retrieval-backed intents, a query parser extracts structured constraints from natural language, TF-IDF or CLIP retrieves 30 candidates, a ranker applies weighted metadata scoring, and then Claude Haiku judges the shortlist for relevance — labeling each product as an exact match, close alternative, or reject. Claude then composes the final response grounded in the accepted set. The key design choice is that retrieval and ranking are deterministic code; the LLM is used only where semantic judgment is actually required. Everything else runs without an API call."

### Talk Track 2: "What was the hardest engineering problem?"

"There were several, but the most interesting was the Railway healthcheck failure. The CLIP model — about 450 MB in memory — was loading synchronously in the FastAPI lifespan hook before `yield`. FastAPI only registers routes after `yield`, so the `/health` endpoint didn't exist yet during model load. Railway's healthcheck timed out and killed the replica in a restart loop. The fix had two parts: baking the model weights into the Docker image layer at build time so the container loads from local cache instead of downloading 600 MB from Hugging Face, and moving the in-memory load to a background daemon thread so the app yields immediately after catalog load. `/health` starts responding in under a second, and image search becomes available ~30 seconds later when the background thread finishes."

### Talk Track 3: "Why the no-fallback rule?"

"Early in the build, I had the orchestrator return 'best available' products if the judge was uncertain. The problem: there's no reliable signal to the user that these are degraded results. Product cards look the same whether they passed a strict judge or were returned as a fallback. This creates a system that appears to work when it's actually broken. The rule I settled on: if the LLM judge or orchestrator is unavailable, the response is an explicit error message and zero products. This applies even if TF-IDF retrieval worked fine. The retrieval result without LLM judgment is not a finished product recommendation — it's an intermediate artifact. Returning it as if it were a finished response is dishonest. I think this is especially important in a demo context, where the evaluator should be able to trust that what they're seeing is what the system is actually capable of."
