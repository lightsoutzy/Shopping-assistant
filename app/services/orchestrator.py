"""
Orchestrator: the single layer responsible for composing user-facing responses
for all retrieval-backed turns (text search, image search, followup on results).

Workers (planner, query_parser, retriever, judge) return structured data.
The orchestrator receives all worker outputs in a ToolBundle and writes the final reply.
This is the ONLY layer that writes user-facing text for search/followup turns.

Design contract (no fallbacks):
  - LLM available  → high-quality response
  - LLM unavailable → raises LLMUnavailableError; caller returns an explicit error message
  There is NO degraded-quality fallback recommendation path.
"""

from dataclasses import dataclass, field
from typing import Optional

from app.schemas import ProductItem
from app.services.judge import JudgeResult


# ── Error type ─────────────────────────────────────────────────────────────────

class LLMUnavailableError(Exception):
    """Raised when the orchestrator cannot call the LLM (no key, timeout, provider error)."""
    pass


# Shown to the user verbatim when the LLM path fails.
LLM_UNAVAILABLE_MSG = (
    "The reasoning model is currently unavailable, so I can't reliably answer this right now. "
    "Please try again in a moment."
)


# ── Data bundle ────────────────────────────────────────────────────────────────

@dataclass
class ToolBundle:
    """All worker outputs for one conversational turn, passed to the orchestrator."""
    # Input context
    user_message: str
    action: str                          # planner hint: new_search | refine_search | followup_on_results | image_search | next_result
    chat_history: list[dict]             # recent transcript [{role, content}, ...]
    search_state_summary: str            # human-readable active filters
    active_products: list[dict]          # products currently in conversation context

    # Worker outputs
    shown_products: list[ProductItem] = field(default_factory=list)   # products being returned this turn
    n_exact: int = 0
    n_close: int = 0
    judge_result: Optional[JudgeResult] = None
    image_dominant_color: str = ""
    filter_context: str = ""             # extra context for filter/followup cases

    # Extracted constraints (for fact-driven response wording)
    requested_count: Optional[int] = None   # user asked for N items
    brand: Optional[str] = None             # specific brand requested
    min_price: Optional[float] = None       # "over $X"
    max_price: Optional[float] = None       # "under $X"


# ── Orchestrator prompt ────────────────────────────────────────────────────────

_ORCHESTRATOR_SYSTEM = """\
You are a warm, knowledgeable fashion shopping assistant for a demo e-commerce store.
You have just run a catalog search and received structured results from your tools.
Your job: write the final conversational reply the user will read.

CRITICAL — speak from the actual facts in the context:
- "Search constraints" block tells you what price range, brand, quantity the user asked for
- "Catalog assessment" block tells you what was actually found
- "Products shown" list tells you exactly what you are showing

Wording rules based on outcome:
- If 1+ exact matches found AND they satisfy the core request: say so positively
  → "I found 2 shoes over $100." / "Yes — I found a Lee Cooper sneaker." / "Here's one that fits."
- If user asked for N items (requested_count) and you found fewer: be honest
  → "I found 2 shoes over $100, not the 3 you asked for."
- If only close alternatives (no exact): explain the specific gap
  → "White hoodies aren't in this catalog right now, but these come closest..."
- If 0 results: be specific about what was searched and what the catalog has
  → "I don't see any Lee Cooper items in this catalog." / "Nothing over $100 in the shoe section currently."
- For follow-up on current results: acknowledge what was filtered or what remains
- For image search: mention visual similarity naturally
- For brand-only queries: treat the brand name as the search intent

Style rules:
- 2–4 sentences, conversational and warm
- NEVER say "I couldn't find an exact match" when exact matches ARE shown
- NEVER say "Here are some products that match" as an opener
- Do NOT start with "Here are"
- Do NOT invent product details not given to you
- Always name specific constraints (price floor, brand name, count) rather than generic phrasing
- Use the search constraints block for accuracy — if min_price is $100, say "over $100" not "over a hundred"
"""


# ── Context builder ────────────────────────────────────────────────────────────

def _build_orchestrator_context(bundle: ToolBundle) -> str:
    lines = []

    # Recent conversation
    if bundle.chat_history:
        lines.append("Recent conversation:")
        for m in bundle.chat_history[-10:]:
            role = str(m.get("role", "user")).capitalize()
            content = str(m.get("content", ""))[:300]
            lines.append(f"  {role}: {content}")
        lines.append("")

    # Action context
    action_labels = {
        "new_search": "New product search",
        "refine_search": "Refined search (user narrowed criteria)",
        "followup_on_results": "Follow-up on currently shown results",
        "image_search": "Visual similarity search (user uploaded an image)",
        "next_result": "User asked to see another item from the same search",
        "bundle_search": "Bundle/combo request — showing multiple complete outfits",
    }
    action_label = action_labels.get(bundle.action, bundle.action)
    lines.append(f"Action: {action_label}")
    lines.append(f'User message: "{bundle.user_message}"')
    lines.append(f"Active search context: {bundle.search_state_summary}")

    if bundle.image_dominant_color:
        lines.append(f"Image dominant color: {bundle.image_dominant_color}")

    if bundle.filter_context:
        lines.append(f"Filter context: {bundle.filter_context}")

    # Explicit search constraints block — gives the LLM ground truth to speak from
    constraint_parts = []
    if bundle.brand:
        constraint_parts.append(f"Brand requested: {bundle.brand}")
    if bundle.min_price is not None:
        constraint_parts.append(f"Minimum price: ${bundle.min_price:.0f}")
    if bundle.max_price is not None:
        constraint_parts.append(f"Maximum price: ${bundle.max_price:.0f}")
    if bundle.requested_count is not None:
        constraint_parts.append(f"Number of results requested by user: {bundle.requested_count}")
    if constraint_parts:
        lines.append("")
        lines.append("Search constraints:")
        for c in constraint_parts:
            lines.append(f"  {c}")

    lines.append("")
    lines.append("Catalog assessment:")
    lines.append(f"  Exact matches: {bundle.n_exact}")
    lines.append(f"  Close alternatives: {bundle.n_close}")
    if bundle.judge_result:
        lines.append(f"  Judge summary: {bundle.judge_result.overall_summary}")
    lines.append("")

    # Products being shown this turn
    if bundle.shown_products:
        lines.append("Products shown to user this turn:")
        for p in bundle.shown_products:
            meta = f"{p.category} | {p.base_color or 'N/A'} | ${p.price:.2f}"
            if p.brand:
                meta += f" | {p.brand}"
            lines.append(f"  - {p.product_name} ({meta})")
    else:
        lines.append("Products shown to user this turn: none")

    # Active context products (relevant for followup narration)
    if bundle.active_products and bundle.action == "followup_on_results":
        lines.append("")
        lines.append("Full current results in context:")
        for p in bundle.active_products[:5]:
            lines.append(
                f"  - {p.get('product_name', '?')} "
                f"({p.get('category', '?')} | {p.get('base_color', 'N/A')} | ${p.get('price', 0):.2f})"
            )

    return "\n".join(lines)


# ── Public API ─────────────────────────────────────────────────────────────────

def compose_turn(bundle: ToolBundle, api_key: str, temperature: float = 0.6) -> str:
    """
    The orchestrator entry point.
    Takes all worker outputs and writes the final user-facing reply.
    Used for all retrieval-backed turns: text search, image search, followup, next-result.

    Raises LLMUnavailableError if the LLM cannot be called.
    There is NO fallback — the caller must handle the error and return an explicit error message.
    """
    if not api_key:
        print(
            f"[ORCHESTRATOR] unavailable action={bundle.action} reason=no_api_key",
            flush=True,
        )
        raise LLMUnavailableError("no_api_key")

    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        context = _build_orchestrator_context(bundle)

        resp = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=384,
            temperature=temperature,
            system=_ORCHESTRATOR_SYSTEM,
            messages=[{"role": "user", "content": context}],
        )
        result = resp.content[0].text.strip()
        print(
            f"[ORCHESTRATOR] source=LLM action={bundle.action} "
            f"exact={bundle.n_exact} close={bundle.n_close}",
            flush=True,
        )
        return result

    except LLMUnavailableError:
        raise
    except Exception as exc:
        print(
            f"[ORCHESTRATOR] unavailable action={bundle.action} reason=provider_error exc={exc!r}",
            flush=True,
        )
        raise LLMUnavailableError(f"provider_error: {exc}") from exc
