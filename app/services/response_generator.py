"""
Generates the final natural language response using Claude.
Falls back to a template response if no API key is configured.

Covers: general chat, clarification, scope refusal.
All search/recommendation responses are handled by orchestrator.py.
"""

from typing import Optional

_ASSISTANT_SYSTEM = """\
You are a helpful fashion shopping assistant for a demo e-commerce store.
You can recommend clothing and accessories, explain what you can do, and help users find products.
You only assist with fashion-related topics. Do not answer unrelated questions.
Keep replies concise and friendly (2–4 sentences max for chat, one paragraph for recommendations).
"""

_CLARIFY_MESSAGE = (
    "I'd love to help you find something! Could you tell me more? "
    "For example: what type of item are you looking for, any color preference, "
    "or what you'll use it for?"
)

_CLARIFY_SYSTEM = """\
You are a fashion shopping assistant. The user's request is unclear or too vague.
Ask one concise, friendly clarification question to understand what they're looking for.
Focus on the most important missing detail (category, color, occasion, or budget).
Keep it to one sentence. Do not list multiple questions."""

_SCOPE_SYSTEM = """\
You are a fashion shopping assistant — you only help with clothing, shoes, bags, and accessories
from the catalog. The user has asked about something outside that scope.
Politely acknowledge you can't help with that, and in one sentence invite them to ask
about fashion items instead. Keep it friendly and brief (1–2 sentences)."""

_SCOPE_REFUSAL = (
    "I'm a fashion shopping assistant — I can only help with products in our catalog. "
    "I can find clothes and accessories, compare current results, or search by image. "
    "What would you like to shop for?"
)


def _fallback_chat(text: str) -> str:
    t = text.lower()
    if any(w in t for w in ("name", "who are you", "what are you")):
        return (
            "I'm your fashion shopping assistant! I can help you find clothing and accessories "
            "from our catalog. Try asking me to recommend a t-shirt, jacket, or bag."
        )
    if any(w in t for w in ("what can", "how do", "help")):
        return (
            "I can help you find fashion products in three ways: "
            "(1) Text search — describe what you want, like 'black casual sneakers under $80'. "
            "(2) Image search — upload a photo and I'll find visually similar items. "
            "(3) General questions about our catalog."
        )
    return (
        "Hi! I'm a fashion shopping assistant. Ask me to recommend a product, "
        "upload an image to find similar items, or ask what I can do."
    )


# ── Public API ─────────────────────────────────────────────────────────────────

def generate_chat(text: str, api_key: str = "") -> str:
    if not api_key:
        return _fallback_chat(text)
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        resp = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=256,
            system=_ASSISTANT_SYSTEM,
            messages=[{"role": "user", "content": text}],
        )
        return resp.content[0].text.strip()
    except Exception:
        return _fallback_chat(text)


def generate_clarification(user_text: str = "", api_key: str = "") -> str:
    """Ask the user a targeted clarification question, LLM-authored when possible."""
    if not api_key:
        return _CLARIFY_MESSAGE
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        resp = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=128,
            system=_CLARIFY_SYSTEM,
            messages=[{"role": "user", "content": user_text or "I want something"}],
        )
        print("[RESPONSE] source=LLM action=clarify_request", flush=True)
        return resp.content[0].text.strip()
    except Exception:
        print("[RESPONSE] source=fallback action=clarify_request", flush=True)
        return _CLARIFY_MESSAGE


def generate_scope_refusal(user_text: str = "", api_key: str = "") -> str:
    """Politely redirect out-of-scope requests, LLM-authored when possible."""
    if not api_key:
        return _SCOPE_REFUSAL
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        resp = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=128,
            system=_SCOPE_SYSTEM,
            messages=[{"role": "user", "content": user_text or "out of scope request"}],
        )
        print("[RESPONSE] source=LLM action=out_of_scope", flush=True)
        return resp.content[0].text.strip()
    except Exception:
        print("[RESPONSE] source=fallback action=out_of_scope", flush=True)
        return _SCOPE_REFUSAL
