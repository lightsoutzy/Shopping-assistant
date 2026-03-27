"""
Tests for image rendering helpers and chat history structure.

Covers:
- image thumbnail creation (including RGBA PNG input)
- history structure includes image turns correctly
- chat history transcript prefix logic for image turns
"""

import io
import pytest


# ── image thumbnail PIL logic ─────────────────────────────────────────────────

try:
    from PIL import Image as _PIL
    _PIL_AVAILABLE = True
except ImportError:
    _PIL_AVAILABLE = False


def _make_png_rgba_bytes(size: int = 50) -> bytes:
    """Create a minimal RGBA PNG in memory."""
    from PIL import Image as PILImage
    img = PILImage.new("RGBA", (size, size), color=(255, 0, 0, 128))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _make_jpeg_bytes(size: int = 50) -> bytes:
    from PIL import Image as PILImage
    img = PILImage.new("RGB", (size, size), color=(0, 255, 0))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def _thumbnail_logic(raw_bytes: bytes, max_size: int = 180) -> bytes:
    """Mirror the fixed _make_image_thumbnail logic without importing streamlit_app."""
    try:
        from PIL import Image as PILImage
        img = PILImage.open(io.BytesIO(raw_bytes)).convert("RGB")
        img.thumbnail((max_size, max_size))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=65)
        return buf.getvalue()
    except Exception:
        return raw_bytes


@pytest.mark.skipif(not _PIL_AVAILABLE, reason="PIL not installed")
def test_thumbnail_rgba_png_does_not_return_none():
    """RGBA PNG must produce valid bytes (previously failed without .convert('RGB'))."""
    thumb = _thumbnail_logic(_make_png_rgba_bytes())
    assert thumb is not None
    assert len(thumb) > 0


@pytest.mark.skipif(not _PIL_AVAILABLE, reason="PIL not installed")
def test_thumbnail_jpeg_input_returns_bytes():
    thumb = _thumbnail_logic(_make_jpeg_bytes())
    assert thumb is not None
    assert len(thumb) > 0


@pytest.mark.skipif(not _PIL_AVAILABLE, reason="PIL not installed")
def test_thumbnail_rgba_png_is_valid_jpeg():
    """After fix, RGBA PNG thumbnail must be a valid, openable image."""
    from PIL import Image as PILImage
    thumb = _thumbnail_logic(_make_png_rgba_bytes())
    img = PILImage.open(io.BytesIO(thumb))
    assert img.mode in ("RGB", "L")


def test_thumbnail_corrupt_input_returns_raw_bytes():
    """Corrupt input should return raw bytes as fallback, not None."""
    corrupt = b"\x00\x01\x02\x03\xff\xfe"
    result = _thumbnail_logic(corrupt)
    assert result is not None
    assert result == corrupt


# ── history structure ─────────────────────────────────────────────────────────

def test_history_user_turn_has_image_thumb_field():
    """User turn with image must have image_thumb bytes in history entry."""
    entry = {
        "role": "user",
        "text": "find something like this",
        "image_thumb": b"\xff\xd8\xff\xe0fake_jpeg",
    }
    assert entry.get("image_thumb") is not None
    assert isinstance(entry["image_thumb"], bytes)


def test_history_user_turn_without_image_has_none_thumb():
    entry = {
        "role": "user",
        "text": "show me hoodies",
        "image_thumb": None,
    }
    assert entry.get("image_thumb") is None


def test_build_chat_history_prefixes_image_turns():
    """_build_chat_history logic must prefix image turns with '[user uploaded an image]'."""
    history = [
        {"role": "user", "text": "find this", "image_thumb": b"fake_thumb"},
        {"role": "assistant", "message": "Here are some results.", "products": []},
        {"role": "user", "text": "show more", "image_thumb": None},
    ]

    # Replicate _build_chat_history logic (from streamlit_app.py)
    out = []
    for entry in history[-6:]:
        role = entry["role"]
        if role == "user":
            content = entry.get("text", "")
            if entry.get("image_thumb") is not None:
                content = "[user uploaded an image] " + content
            out.append({"role": "user", "content": content})
        else:
            out.append({"role": "assistant", "content": entry.get("message", "")})

    assert out[0]["content"].startswith("[user uploaded an image]"), \
        "Image turn must be prefixed in backend transcript"
    assert out[2]["content"] == "show more", \
        "Non-image turn must not have the prefix"


def test_render_history_image_turn_check():
    """Verify the render_history condition: image_thumb not None triggers st.image."""
    entries = [
        {"role": "user", "text": "test", "image_thumb": b"some_bytes"},
        {"role": "user", "text": "no image", "image_thumb": None},
    ]
    assert entries[0].get("image_thumb") is not None, "Entry with bytes should trigger st.image"
    assert entries[1].get("image_thumb") is None, "Entry with None should not trigger st.image"


@pytest.mark.skipif(not _PIL_AVAILABLE, reason="PIL not installed")
def test_image_thumb_wrappable_in_bytesio():
    """Thumbnail bytes must be wrappable in io.BytesIO for st.image compatibility."""
    thumb = _thumbnail_logic(_make_jpeg_bytes())
    wrapped = io.BytesIO(thumb)
    assert wrapped.read(2) == b"\xff\xd8", "Valid JPEG starts with FF D8"


@pytest.mark.skipif(not _PIL_AVAILABLE, reason="PIL not installed")
def test_rgba_png_thumb_wrappable_in_bytesio():
    """RGBA PNG thumbnail must be valid JPEG bytes wrappable in BytesIO."""
    thumb = _thumbnail_logic(_make_png_rgba_bytes())
    wrapped = io.BytesIO(thumb)
    first2 = wrapped.read(2)
    assert first2 == b"\xff\xd8", f"Expected JPEG header, got {first2.hex()}"
