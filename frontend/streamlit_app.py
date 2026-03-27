"""
Fashion Shopping Assistant — Streamlit frontend.
Calls POST /agent on the FastAPI backend.

History format (list of dicts):
  User turn:      {"role": "user", "text": str, "image_thumb": bytes | None}
  Assistant turn: {"role": "assistant", "message": str, "products": list,
                   "eval_passed": bool | None}
"""

import base64
import io
import os
import re
import uuid
from pathlib import Path

import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

API_BASE = (
    os.getenv("API_URL")
    or "http://{}:{}".format(os.getenv("API_HOST", "localhost"), os.getenv("API_PORT", "8000"))
).rstrip("/")
DATASET_ROOT = Path(os.getenv("DATASET_ROOT", "data/fashion-dataset/fashion-dataset"))

# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Fashion Shopping Assistant",
    page_icon="👗",
    layout="wide",
)

# ── session state ─────────────────────────────────────────────────────────────
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "history" not in st.session_state:
    # Each entry: {"role": "user"|"assistant", ...}
    st.session_state.history = []
if "active_products" not in st.session_state:
    st.session_state.active_products = []
if "last_image_hash" not in st.session_state:
    st.session_state.last_image_hash = None
if "search_state" not in st.session_state:
    st.session_state.search_state = {}
if "pending_suggestion" not in st.session_state:
    st.session_state.pending_suggestion = None
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0


# ── text rendering helpers ────────────────────────────────────────────────────

def _safe_render(text: str) -> str:
    """
    Prevent dollar sign pairs like '$50 and $75' from being parsed as LaTeX
    inline math by Streamlit's markdown renderer.
    Insert a zero-width space (U+200B) after each '$' that precedes a digit.
    Renders visually identical but breaks the $..$ LaTeX delimiter pattern.
    """
    return re.sub(r'\$(\d)', '$\u200b\\1', text)


# ── image helpers ─────────────────────────────────────────────────────────────

def _make_image_thumbnail(raw_bytes: bytes, max_size: int = 180) -> bytes:
    """
    Return a small JPEG thumbnail for display in chat history.
    Falls back to raw bytes so the image is always displayable.
    """
    try:
        from PIL import Image as PILImage
        img = PILImage.open(io.BytesIO(raw_bytes)).convert("RGB")  # handles RGBA PNG
        img.thumbnail((max_size, max_size))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=65)
        return buf.getvalue()
    except Exception:
        # PIL unavailable or corrupt image — return raw bytes (Streamlit handles PNG/JPEG)
        return raw_bytes


# ── chat history builder ──────────────────────────────────────────────────────

def _build_chat_history() -> list[dict]:
    """
    Return last 6 entries as [{role, content}] for backend context.
    Image turns are represented as "[user uploaded an image] <text>".
    We do NOT re-send image bytes — just a marker in the transcript.
    """
    out = []
    for entry in st.session_state.history[-12:]:
        role = entry["role"]
        if role == "user":
            content = entry.get("text", "")
            if entry.get("image_thumb") is not None:
                content = "[user uploaded an image] " + content
            out.append({"role": "user", "content": content})
        else:
            out.append({"role": "assistant", "content": entry.get("message", "")})
    return out


# ── backend call ──────────────────────────────────────────────────────────────

def call_agent(text: str, image_b64: str | None) -> dict:
    payload = {
        "text": text,
        "image_b64": image_b64,
        "session_id": st.session_state.session_id,
        "chat_history": _build_chat_history(),
        "active_products": st.session_state.active_products,
        "search_state": st.session_state.search_state,
    }
    resp = requests.post(f"{API_BASE}/agent", json=payload, timeout=60)
    resp.raise_for_status()
    return resp.json()


# ── product rendering ─────────────────────────────────────────────────────────

def _image_source(product_id: int, image_path: str):
    """Return a URL (deployed) or local path string (local dev), or None."""
    if os.getenv("API_URL"):
        return f"{API_BASE}/images/{product_id}.jpg"
    local = DATASET_ROOT / image_path
    return str(local) if local.exists() else None


def _render_product_card(col, p: dict) -> None:
    """Render a single product card inside a column."""
    with col:
        src = _image_source(p["id"], p["image_path"])
        if src:
            st.image(src)
        else:
            st.markdown("*(no image)*")
        st.markdown(f"**{p['product_name']}**")
        meta_parts = [p["category"]]
        if p.get("base_color"):
            meta_parts.append(p["base_color"])
        if p.get("usage"):
            meta_parts.append(p["usage"])
        st.caption(" · ".join(meta_parts))
        st.markdown(f"**\\${p['price']:.2f}**")
        if p.get("brand"):
            st.caption(p["brand"])
        if p.get("reason"):
            st.caption(p["reason"])


def render_products(products: list[dict]) -> None:
    if not products:
        return
    cols = st.columns(min(len(products), 5))
    for col, p in zip(cols, products):
        _render_product_card(col, p)


def render_bundles(bundle_pairs: list[list[dict]]) -> None:
    """Render 2-item bundle pairs as grouped cards with total price."""
    if not bundle_pairs:
        return
    for i, bundle in enumerate(bundle_pairs):
        total = sum(p.get("price", 0) for p in bundle)
        st.markdown(f"**Bundle {i + 1}** — Total: **\\${total:.2f}**")
        cols = st.columns(len(bundle))
        for col, p in zip(cols, bundle):
            _render_product_card(col, p)
        if i < len(bundle_pairs) - 1:
            st.divider()


# ── history rendering ─────────────────────────────────────────────────────────

def render_history() -> None:
    for entry in st.session_state.history:
        role = entry["role"]
        if role == "user":
            with st.chat_message("user"):
                # Show uploaded image thumbnail if this turn had one.
                # Wrap in BytesIO — st.image requires BytesIO, not raw bytes.
                thumb = entry.get("image_thumb")
                if thumb is not None:
                    st.image(io.BytesIO(thumb), width=150, caption="Uploaded image")
                st.write(entry.get("text", ""))
        else:
            with st.chat_message("assistant"):
                st.markdown(_safe_render(entry.get("message", "")))
                if entry.get("bundle_pairs"):
                    render_bundles(entry["bundle_pairs"])
                else:
                    render_products(entry.get("products", []))


# ── sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("Fashion Assistant")
    st.markdown("A demo shopping agent powered by Claude + CLIP.")
    st.divider()

    st.subheader("Image search")
    uploaded_file = st.file_uploader(
        "Upload a product image",
        type=["jpg", "jpeg", "png"],
        key=f"uploader_{st.session_state.uploader_key}",
    )
    st.caption("Upload an image and send a message to find visually similar products.")

    st.divider()

    # Read-only current search state summary
    ss = st.session_state.search_state
    filter_parts = []
    if ss.get("category"):   filter_parts.append(ss["category"])
    if ss.get("color"):      filter_parts.append(ss["color"])
    if ss.get("max_price"):  filter_parts.append(f"≤${ss['max_price']:.0f}")
    if ss.get("gender"):     filter_parts.append(ss["gender"])
    if ss.get("usage"):      filter_parts.append(ss["usage"])
    if ss.get("plain_only"): filter_parts.append("plain")
    if ss.get("image_active"): filter_parts.append("📷 image")
    if filter_parts:
        st.subheader("Current search")
        st.caption(" · ".join(filter_parts))
        result_count = ss.get("result_count", 0)
        had_perfect = ss.get("had_perfect_matches")
        if result_count:
            quality = "exact" if had_perfect else "close"
            st.caption(f"{result_count} {quality} match(es) shown")
        st.caption("Say **'start over'** to clear.")
        st.divider()

    if st.button("Clear conversation"):
        st.session_state.history = []
        st.session_state.active_products = []
        st.session_state.last_image_hash = None
        st.session_state.search_state = {}
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.uploader_key += 1  # clear any lingering uploaded image
        st.rerun()

    st.caption(f"Session: `{st.session_state.session_id[:8]}…`")


# ── main area ─────────────────────────────────────────────────────────────────
st.header("Fashion Shopping Assistant")
st.markdown(
    "A scoped shopping assistant for catalog discovery, product comparison, and image-based search. "
    "**It only answers questions about products in this catalog** — not general questions."
)

# Suggestion chips — only shown before the first user turn.
# Only include prompts that make sense on an empty conversation (no follow-ups).
_SUGGESTIONS = [
    "Recommend me a black hoodie for casual wear",
    "Show me white sneakers under $80",
    "Find me a winter jacket for men under $100",
    "Suggest a hoodie and sneaker combo under $150",
]

if not st.session_state.history:
    st.markdown("##### Try asking:")
    _cols = st.columns(len(_SUGGESTIONS))
    for _col, _sug in zip(_cols, _SUGGESTIONS):
        with _col:
            if st.button(_sug, use_container_width=True, key=f"sug_{_sug[:20]}"):
                st.session_state.pending_suggestion = _sug
    st.divider()

render_history()

# ── chat input ────────────────────────────────────────────────────────────────
user_input = st.chat_input("Ask something, e.g. 'Find me a black casual jacket under $100'")

if not user_input and st.session_state.pending_suggestion:
    user_input = st.session_state.pending_suggestion
    st.session_state.pending_suggestion = None

if user_input:
    # ── process image upload ──────────────────────────────────────────────────
    image_b64 = None
    image_thumb = None
    _current_image_hash = None
    _sent_image = False  # tracks whether this turn included an image

    if uploaded_file is not None:
        import hashlib
        _raw_bytes = uploaded_file.read()
        _current_image_hash = hashlib.md5(_raw_bytes).hexdigest()
        if _current_image_hash != st.session_state.last_image_hash:
            # New image not yet submitted — send it now and create thumbnail
            image_b64 = base64.b64encode(_raw_bytes).decode("utf-8")
            image_thumb = _make_image_thumbnail(_raw_bytes)

    # ── append user turn to history ───────────────────────────────────────────
    st.session_state.history.append({
        "role": "user",
        "text": user_input,
        "image_thumb": image_thumb,  # None if no image this turn
    })

    # ── render user turn immediately ──────────────────────────────────────────
    with st.chat_message("user"):
        if image_thumb is not None:
            st.image(io.BytesIO(image_thumb), width=150, caption="Uploaded image")
        st.write(user_input)

    # ── call backend and render response ─────────────────────────────────────
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                result = call_agent(text=user_input, image_b64=image_b64)
                st.markdown(_safe_render(result["message"]))
                bundle_pairs = result.get("bundle_pairs", [])
                if bundle_pairs:
                    render_bundles(bundle_pairs)
                else:
                    render_products(result.get("products", []))

                # Mark image as consumed and schedule uploader clear
                if image_b64 and _current_image_hash:
                    st.session_state.last_image_hash = _current_image_hash
                    st.session_state.uploader_key += 1  # new key → widget remounts empty
                    _sent_image = True

                # Update active_products
                context_action = result.get("context_action", "keep")
                if context_action == "replace":
                    st.session_state.active_products = result.get("products", [])
                elif context_action == "clear":
                    st.session_state.active_products = []

                # Persist search_state
                if result.get("search_state"):
                    st.session_state.search_state = result["search_state"]

                # Append assistant turn to history
                st.session_state.history.append({
                    "role": "assistant",
                    "message": result["message"],
                    "products": result.get("products", []),
                    "bundle_pairs": bundle_pairs,
                    "eval_passed": result.get("eval_passed"),
                })

            except requests.exceptions.ConnectionError:
                st.error(f"Cannot reach the backend at `{API_BASE}`. Is FastAPI running?")
                # Remove the optimistically-added user turn on failure
                if st.session_state.history and st.session_state.history[-1]["role"] == "user":
                    st.session_state.history.pop()
            except requests.exceptions.HTTPError as e:
                st.error(f"Backend error: {e}")
                if st.session_state.history and st.session_state.history[-1]["role"] == "user":
                    st.session_state.history.pop()
            except Exception as e:
                st.error(f"Unexpected error: {e}")
                if st.session_state.history and st.session_state.history[-1]["role"] == "user":
                    st.session_state.history.pop()

    # After the turn is fully rendered and state is committed, force a rerun so the
    # uploader widget remounts with the new key (visually clearing the uploaded file).
    if _sent_image:
        st.rerun()
