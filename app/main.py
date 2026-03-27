import threading
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.data.catalog_loader import load_catalog
from app.services.retriever import TfidfRetriever


def _load_clip_background(app, store):
    """Load CLIP model in a background thread so /health is available immediately."""
    try:
        from app.services.image_search import build_image_index, load_clip_model
        print("Loading CLIP model for image search (background) ...", flush=True)
        processor, model = load_clip_model()
        app.state.image_index = build_image_index(
            store.embeddings, store.embedding_ids, store.df
        )
        app.state.clip_processor = processor
        app.state.clip_model = model
        print("  Image search ready.", flush=True)
    except Exception as e:
        print(f"  Warning: image search unavailable ({e}).", flush=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── startup ───────────────────────────────────────────────────────────────
    from app import config as _cfg
    if _cfg.ANTHROPIC_API_KEY:
        key_hint = f"{_cfg.ANTHROPIC_API_KEY[:8]}…{_cfg.ANTHROPIC_API_KEY[-4:]}"
        print(f"[CONFIG] ANTHROPIC_API_KEY loaded ({key_hint}) — LLM enabled", flush=True)
    else:
        print("[CONFIG] ANTHROPIC_API_KEY missing — judge/orchestrator will return errors", flush=True)
        print("  → Add ANTHROPIC_API_KEY to .env (or .env.txt on Windows)", flush=True)

    print("Loading catalog ...")
    store = load_catalog()
    app.state.catalog = store
    app.state.retriever = TfidfRetriever(store.df)
    print(f"  {len(store.df):,} products indexed.")

    # ── image search (optional; skipped if embeddings not built yet) ──────────
    app.state.image_index = None
    app.state.clip_processor = None
    app.state.clip_model = None

    if store.embeddings is not None:
        threading.Thread(
            target=_load_clip_background, args=(app, store), daemon=True
        ).start()

    yield  # /health available immediately; CLIP loads in background

    # ── shutdown (nothing to release) ─────────────────────────────────────────


app = FastAPI(
    title="Fashion Shopping Assistant",
    description="Unified agent endpoint for a fashion e-commerce demo.",
    version="0.1.0",
    lifespan=lifespan,
)

from app.api.routes_agent import router  # noqa: E402  (import after app creation)

app.include_router(router)

_thumb_dir = Path("data/catalog_thumbnails")
if _thumb_dir.exists():
    app.mount("/images", StaticFiles(directory=str(_thumb_dir)), name="images")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/diagnostics")
def diagnostics():
    """
    Safe startup diagnostics — performs a live Anthropic ping to confirm the key works.
    Never exposes the raw API key.
    """
    from app import config as _cfg
    key = _cfg.ANTHROPIC_API_KEY
    key_status = f"present ({key[:8]}…{key[-4:]})" if key else "missing"
    catalog_ok = _cfg.CATALOG_PATH.exists()
    image_ready = app.state.image_index is not None

    # Live Anthropic ping — catches auth errors, quota issues, wrong key, network problems
    llm_ping = "not_tested"
    llm_error = None
    if key:
        try:
            import anthropic as _anthropic
            _client = _anthropic.Anthropic(api_key=key)
            _resp = _client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=5,
                messages=[{"role": "user", "content": "ping"}],
            )
            llm_ping = "ok"
        except Exception as exc:
            llm_ping = "failed"
            llm_error = f"{type(exc).__name__}: {str(exc)[:200]}"
    else:
        llm_ping = "skipped_no_key"

    return {
        "anthropic_api_key": key_status,
        "llm_ping": llm_ping,
        "llm_error": llm_error,
        "catalog_loaded": catalog_ok,
        "image_search_ready": image_ready,
        "model": "claude-haiku-4-5-20251001",
        "env_file": str(_cfg._env if _cfg._env.exists() else (_cfg._env_txt if _cfg._env_txt.exists() else "none")),
    }
