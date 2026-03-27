# Development Journey

## What Worked Early

The core retrieval pipeline came together quickly. TF-IDF over a preprocessed parquet file is a well-understood pattern: load the catalog, fit a vectorizer, expose a search function. The metadata filtering logic (category, color, gender, usage, price) was straightforward to implement as hard pre-filters before the TF-IDF ranking step. The first end-to-end text query — "black casual shirt for men" returning relevant products — worked on the first full pipeline run.

The Streamlit frontend similarly came together fast. Streamlit's chat primitives (`st.chat_input`, `st.chat_message`, session state) mapped directly to the required UX. Product cards in columns, image display via `st.image`, and the sidebar uploader were all standard Streamlit patterns.

The planner's rule-based cascade was effective from the start. Most queries route correctly with regex patterns alone. The LLM fallback for ambiguous cases handles the long tail without adding latency to the common path.

## What Failed and Was Fixed

### Removing Fallback Recommendation Logic

An early version of the orchestrator would return "best available" products when the judge was uncertain or the LLM returned a weak response. This was removed deliberately. The problem: a system that silently degrades its quality criteria is indistinguishable from one that is working correctly. Users see product cards either way. The fix was enforcing `JudgeUnavailableError` and `LLMUnavailableError` as explicit exceptions that the endpoint catches and converts to user-visible error messages. Zero products is the correct response when the pipeline is broken.

### Image Uploader Reset Bug

After submitting an image query, the Streamlit file uploader retained the uploaded image. The next user message would re-send the same image, triggering another image search instead of a text follow-up. The fix was tracking the uploaded image's SHA-256 hash in session state (`last_image_hash`). On submission, the image is consumed: the hash is updated, the uploader is remounted (by incrementing `uploader_key`), and subsequent text messages send no image.

### .env / Config / Path Issues

On Windows, the `.env` extension is sometimes hidden by Explorer and apps may save files as `.env.txt`. `python-dotenv` only loads `.env` by default. The config module was updated to try `.env` first, then `.env.txt`. A separate issue: relative paths in the config resolved against the working directory rather than the config file's location, which broke when the server was started from a different directory. Paths were made absolute by resolving against `Path(__file__).parent`.

### Python Version and anthropic/httpx Mismatch

The Anthropic Python SDK has a dependency on `httpx`. An early version of the local environment had a pinned `httpx` version that conflicted with the SDK's requirements. This surfaced as a confusing import error. The fix was unpinning `httpx` from `requirements.txt` and letting `pip` resolve it through the `anthropic` dependency.

### Image + Text Multimodal Formatting Fix

The image judge sends query image bytes to Claude's multimodal API. The original code passed raw uploaded bytes directly with `"media_type": "image/jpeg"` hardcoded. When a user uploaded a PNG, the bytes were PNG but the declared media type was JPEG — Claude returned an error, and the endpoint returned a generic failure message to the user.

The fix was re-encoding every uploaded image as JPEG before sending to the judge:
```python
_jpeg_buf = io.BytesIO()
pil_image.save(_jpeg_buf, format="JPEG", quality=85)
query_image_b64_jpeg = base64.b64encode(_jpeg_buf.getvalue()).decode()
```
This also handles RGBA PNGs (transparency channel dropped during JPEG conversion) without any special casing.

### Follow-up Grounding (SearchState Not Persisted)

After a follow-up query, `_handle_followup()` was returning responses without including `search_state=ss` in the `AgentResponse`. The frontend used the default empty `SearchState` from the response, losing all filter context. The next turn would behave as if no prior search had occurred.

The fix was ensuring every return path in `_handle_followup()` included `search_state=ss`. This sounds trivial but the function had five exit paths and only the happy path had been wired correctly.

A related issue: the planner's `_NEW_SEARCH_SIGNAL_RE` guard matched `\brecommend\b`, so "which one would you recommend?" triggered a new search instead of a follow-up. The fix was adding a `_BEST_ONE_RE` pattern at a higher priority in the cascade, before the new-search guard check.

### Railway: Torch CUDA Wheel Size

Railway builds the Docker image from the repo. The default `torch==2.3.0` wheel on Linux pulls CUDA support (~3.5 GB installed). Railway's image size limit is 4 GB. The total image — Python base + torch + other deps + model weights + catalog + thumbnails — exceeded 6 GB.

The fix was pinning to the CPU-only wheel: `torch==2.3.0+cpu` via `--extra-index-url https://download.pytorch.org/whl/cpu`. The CPU build installs in ~800 MB. CLIP inference is CPU-bound at demo scale anyway, so there is no performance regression.

### Railway: Nixpacks Pip Unavailability

After fixing the image size, the Nixpacks build failed with `pip: command not found`. Nixpacks runs Python from the Nix package store, which places it in an environment that lacks the `pip` binary in the standard PATH. Attempting to call `python -m pip` failed with `No module named pip`.

The fix was abandoning Nixpacks and switching to a plain `Dockerfile` with `FROM python:3.11-slim`. The `python:3.11-slim` image always has pip. The `railway.toml` builder was changed to `"dockerfile"`. The `nixpacks.toml` file was deleted.

### Railway: CLIP Startup Blocking the Healthcheck

The CLIP model was loaded synchronously in the FastAPI lifespan hook before `yield`. FastAPI only registers routes (including `/health`) after `yield`. A 30-60 second CLIP load caused Railway's healthcheck to time out (60 seconds default) and kill the replica in a restart loop.

Two fixes applied in sequence:

**Fix 1 — Bake model weights into the Docker image:** A `RUN python -c "CLIPModel.from_pretrained(...)"` step in the Dockerfile downloads the ~600 MB weights at build time and caches them in the image layer. Container startup then loads from local cache in ~2 seconds instead of downloading from Hugging Face.

**Fix 2 — Background thread:** Even with local weights, loading the model into RAM takes 10-30 seconds. The lifespan was changed to start a daemon thread for CLIP loading before `yield`, then yield immediately. `/health` becomes available as soon as the catalog loads (~1 second). Image search responds with "initializing" for the first 30 seconds, then becomes fully operational.

### Streamlit Cloud: Python 3.14 Pillow Dependency Conflicts

Streamlit Community Cloud deployed the app on Python 3.14.3. Three separate dependency issues surfaced:

1. `requirements.txt` still included `torch==2.3.0` (a leftover from the original monolithic requirements). No `torch` wheel exists for Python 3.14. Fix: make `requirements.txt` frontend-only (4 packages).

2. After removing torch, `Pillow==10.3.0` was still listed. No Pillow 10.x wheel exists for Python 3.14 (build from source fails with `KeyError: '__version__'`). Fix: remove explicit Pillow from `requirements.txt` — the `_make_image_thumbnail()` function already has a `try/except` fallback.

3. Even without explicit Pillow, `streamlit==1.33.0` pulls Pillow transitively and pins it to `<11`. Pip tried to build Pillow 10.4.0 from source, which fails on Python 3.14. Fix: upgrade to `streamlit==1.40.0`, which relaxes the bound to `pillow<12`, allowing Pillow 11.x which has Python 3.14 pre-built wheels.

## Key Debugging Milestones

- First full pipeline run: text query → retrieval → judge → orchestrator → structured response
- First image search working end-to-end with correct JPEG re-encoding
- First multimodal judge call with actual product thumbnail images
- First follow-up query maintaining correct filter context across turns
- First bundle query returning paired results with total price
- First Railway deploy passing both build and runtime healthcheck
- First Streamlit Cloud deploy loading successfully without dependency errors
