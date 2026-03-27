# Problem Statement and Constraints

## Problem Statement

Fashion product discovery is search-engine-centric: users type keywords, browse filtered grids, and repeat. This works for users who know exactly what they want but fails for vague intent ("something to wear to a casual dinner"), visual reference ("something like this photo"), or iterative refinement ("now show me the same thing in navy under $70").

The goal was to prototype an AI-native discovery experience: a conversational agent that accepts intent in natural form (text or image), maintains search context across turns, and returns curated results with explanations — rather than a ranked list the user must interpret themselves.

## Imposed and Practical Constraints

**Scope:** Must behave like one unified agent. Internally it routes to different pipelines, but the user should experience one coherent shopping assistant, not three separate tools.

**Demo timeframe:** Built incrementally over roughly 1-2 days of focused work. This forced prioritization: what is the minimum implementation that demonstrates the full capability loop end-to-end?

**API cost:** LLM inference costs real money. The system uses Claude Haiku (cheapest capable model) for parsing and orchestration, and applies per-session (100 calls) and global (2000 calls) rate limits to prevent a public demo from becoming an unbounded expense.

**Deployment budget:** Both Railway (backend) and Streamlit Community Cloud (frontend) are free tiers. This imposed hard constraints: Railway's 4 GB image size limit, no persistent disk, no GPU. Streamlit Cloud's Python version (3.14.3 at time of deployment) imposed its own dependency compatibility requirements.

## Why a Predefined Catalog Was Used

Using the Kaggle Fashion Product Images Dataset as a static catalog was the correct architectural choice for a demo:

- Reproducible: same products, same IDs, same embeddings across rebuilds
- Bounded: LLM cannot hallucinate products that don't exist because the shortlist is always drawn from the catalog
- Testable: query parsing and retrieval can be validated against known product records
- No API dependency: no external retail API key, no rate limits on catalog access

A live inventory API (Shopify, ASOS) would have added auth complexity, flaky test behavior, and cost — all overhead that obscures the architecture rather than demonstrating it.

## Why Hosted Demo Mattered

A deployed URL is the difference between a portfolio piece and a zip file. For an AI Engineer role evaluation, reviewers should be able to interact with the system in a browser without setting up a local environment. The two-service topology (Railway backend + Streamlit Cloud frontend) enables this while keeping the API key server-side only, which is a hard requirement.

## Why No Fallback Recommendation Behavior

This was an explicit design choice made early and enforced throughout the build.

The alternative — returning "best available" products when the judge or orchestrator fails — creates a system that appears to work when it is actually broken. A user sees product cards and assumes the system understood their query. A developer sees a passing healthcheck and misses a silent failure.

The rule: if the LLM judge or orchestrator is unavailable (no API key, network error, rate limit), the endpoint returns an explicit error message and zero products. The UI shows the error message in the chat. The user knows the system is unavailable. This is honest behavior.

Text retrieval (TF-IDF) and metadata filtering still work without the LLM. But those components are not wired to produce a fake "good enough" response — they feed a pipeline that requires LLM judgment to complete.

## Key Deployment Constraints Encountered

**Railway 4 GB image size limit:** The default PyTorch wheel on Linux pulls CUDA support (~3.5 GB installed). The fix was pinning to the CPU-only wheel (`torch==2.3.0+cpu`) via PyTorch's dedicated index URL, cutting the installed size to ~800 MB.

**Nixpacks pip unavailability:** Railway's default Nixpacks builder runs Python from the Nix store, which lacks pip in the expected path. The fix was switching to a plain `Dockerfile` using `python:3.11-slim`, which always has pip.

**CLIP model startup blocking Railway healthcheck:** The CLIP model (~450 MB in memory) was originally loaded synchronously in the FastAPI lifespan hook before `yield`. Since FastAPI only registers routes (including `/health`) after `yield`, a 30–60 second model load caused the healthcheck to time out and Railway killed the replica. The fix was pre-baking model weights into the Docker image layer (eliminating the download) and moving the in-memory load to a background daemon thread, so `yield` is reached immediately after catalog load.

**Streamlit Python 3.14 dependency conflicts:** Streamlit Community Cloud deployed on Python 3.14.3. `streamlit==1.33.0` pins `pillow<11`, and Pillow 10.x has no pre-built wheel for Python 3.14 (build from source fails). The fix was upgrading to `streamlit==1.40.0`, which relaxes the Pillow bound to `<12`, allowing Pillow 11.x which has Python 3.14 wheels.

**Dependency file routing:** Streamlit Cloud unconditionally installs from `requirements.txt`. There is no configuration mechanism to point it at a differently-named file. The fix was making `requirements.txt` frontend-only (4 packages) and leaving all backend/torch dependencies exclusively in `requirements-backend.txt` (used by the Dockerfile).
