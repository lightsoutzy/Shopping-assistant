"""
Planner / controller step.

Determines the action for each turn before any retrieval or generation.
Rule-based for unambiguous cases; LLM for ambiguous middle cases.

Action labels:
  ask_capabilities      - user asking what the assistant can do / what it is
  ask_current_state     - user asking about active filters/search state
  new_search            - fresh product text search with new criteria
  refine_search         - narrow or change constraints on current search
  followup_on_results   - question/sort/compare on currently shown products
  compare_results       - explicit side-by-side comparison
  image_search          - visual similarity search
  explain_match_quality - user asking why results are approximate
  next_result           - user wants next items from accepted pool
  reset_context         - start over
  out_of_scope          - not shopping-related
  bundle_search         - 2-item combo/outfit request (e.g. hoodie + sneaker)
"""

import json
import re
from dataclasses import dataclass
from typing import Optional

# ── Re-use patterns from intent_router for consistency ────────────────────────
from app.services.intent_router import (
    _OUT_OF_SCOPE_RE,
    _RECOMMEND_RE,
    _RESET_RE,
)

# ── Planner-specific patterns ─────────────────────────────────────────────────

_CAPABILITIES_RE = re.compile(
    r"^(hi|hello|hey|howdy)\b"
    r"|\bwhat (is|are|can) you\b"
    r"|\bwho are you\b"
    r"|\byour name\b"
    r"|\bwhat do you do\b"
    r"|\bhow (do|can) (i|you)\b"
    r"|\bhelp me$"
    r"|\bwhat (kind|type|sort) of"
    r"|\bwhat products\b"
    r"|\bcan you help\b"
    r"|\bwhat can you\b",
    re.IGNORECASE,
)

_STATE_QUESTION_RE = re.compile(
    r"\bwhat('s| is) (my |the |our |current |active |)?(filter|search|constraint|criteria|looking for|searching for)\b"
    r"|\bwhat are we (looking for|searching for|searching)\b"
    r"|\bcurrent (filter|search|constraint|criteria)\b"
    r"|\bactive (filter|search)\b"
    r"|\bwhat (did|do) (i|you) (filter|search|ask for|set)\b"
    r"|\bwhat('s| is) (the |my |our )?(search|filter)\b",
    re.IGNORECASE,
)

_EXPLAIN_QUALITY_RE = re.compile(
    r"\bwhy (are|were|is|did) (these|this|they|the results?|you) (only |)(alternatives?|close|approximate|not exact|not perfect)\b"
    r"|\bwhy (not|didn't|don't) (you )?(show|find) (exact|perfect|exact match)s?\b"
    r"|\bdo you have (any )?(exact|perfect) match(es)?\b"
    r"|\bwhy (alternatives?|approximate|close alternatives?)\b"
    r"|\bwhy (only |)(close|approximate)\b",
    re.IGNORECASE,
)

_NEXT_RESULT_RE = re.compile(
    r"\bshow me (another|one more|the next|a different one)\b"
    r"|\bgive me (another|one more|the next|a different one)\b"
    r"|\bsee (another|one more|the next)\b"
    r"|\bone more (option|item|product|result)\b"
    r"|\banother option\b"
    r"|\bnext (one|item|option|result)\b",
    re.IGNORECASE,
)

_COMPARE_RE = re.compile(
    r"\bcompare (these|those|them|the)\b"
    r"|\bversus\b|\bvs\.?\b"
    r"|\bbetween (these|those|them|the)\b"
    r"|\bwhich one should i (buy|get|pick|choose)\b"
    r"|\bdifference between\b",
    re.IGNORECASE,
)

_FOLLOWUP_RE = re.compile(
    r"\b(cheapest|least expensive|most expensive|priciest|lowest price|highest price)\b"
    r"|\bof (these|those|them)\b"
    r"|\bonly show|just show\b"
    r"|\bonly the|just the\b"
    r"|\bbest (for|looking|one|option)\b"
    r"|\bthe \w+ one\b"
    r"|\bones?\b",
    re.IGNORECASE,
)

# Signals that clearly mean a NEW search (action verb + optional item type).
# Used as a narrower guard for followup/refine routing so that bare
# colour/attribute words like "white ones?" don't accidentally block followup.
_NEW_SEARCH_SIGNAL_RE = re.compile(
    r"\b(recommend|suggest|find|show me|look for|search|get me|need|want)\b"
    r"|\b(t-shirt|tshirt|shirt|shoes?|sneakers?|jacket|hoodie|sweatshirt|shorts|bag|handbag|backpack)\b",
    re.IGNORECASE,
)

_BEST_ONE_RE = re.compile(
    r"\bwhich (one|item|of these|of those|would you)\b"
    r"|\bif you (had to|have to) (recommend|pick|choose)\b"
    r"|\bwhich (is|are) (best|closest|cheapest|most versatile)\b"
    r"|\bnarrow (it |this )?down\b"
    r"|\bamong these\b"
    r"|\bwhich one (is|would) (best|closest|cheapest)\b",
    re.IGNORECASE,
)

_BUNDLE_RE = re.compile(
    r"\b(combo|outfit|bundle)\b"
    r"|\b(hoodie|sweatshirt|jacket|shirt|t-shirt|tshirt)\b.{0,60}\b(shoe|sneaker|boot|pant|trouser|jean|short)\b"
    r"|\b(shoe|sneaker|boot|pant|trouser|jean|short)\b.{0,60}\b(hoodie|sweatshirt|jacket|shirt|t-shirt|tshirt)\b",
    re.IGNORECASE,
)

_REFINE_RE = re.compile(
    r"\bonly (plain|solid)\b"
    r"|\bmake it (under|below|less than|max) \$?\d+\b"
    r"|\b(under|below|less than|max) \$?\d+\b"
    r"|\bremove the (price|color|colour|budget|filter|category|limit)\b"
    r"|\bno (price limit|budget limit|filter)\b"
    r"|\bswitch (to|from) \w+\b"
    r"|\bchange (it )?to \w+\b"
    r"|\bonly plain\b"
    r"|\bplain ones?\b"
    r"|\bjust (plain|solid)\b",
    re.IGNORECASE,
)


@dataclass
class PlannerDecision:
    action: str
    reasoning: str  # internal only, never shown to user


def _has_active_search(search_state: dict) -> bool:
    """True if there is an active search context with at least one constraint."""
    return bool(
        search_state.get("category")
        or search_state.get("color")
        or search_state.get("last_query_summary")
        or search_state.get("last_action")
        or search_state.get("last_intent_action")
        or search_state.get("image_active")
    )


def plan(
    user_message: str,
    search_state: dict,
    has_image: bool,
    has_active_products: bool,
    api_key: Optional[str] = None,
    chat_history: Optional[list] = None,
) -> PlannerDecision:
    """
    Determine the action for this turn.
    Rule-based for clear cases; LLM (with transcript) for ambiguous cases.
    """
    text = user_message.strip()

    # ── Priority 1: Image always wins ─────────────────────────────────────────
    if has_image:
        return PlannerDecision("image_search", "image attached")

    # ── Priority 2: Explicit reset ────────────────────────────────────────────
    if _RESET_RE.match(text):
        return PlannerDecision("reset_context", "explicit reset signal")

    # ── Priority 3: Hard out-of-scope ─────────────────────────────────────────
    if _OUT_OF_SCOPE_RE.search(text):
        return PlannerDecision("out_of_scope", "clearly non-shopping topic")

    # ── Priority 4: State question (only meaningful with active search) ────────
    if _STATE_QUESTION_RE.search(text) and _has_active_search(search_state):
        return PlannerDecision("ask_current_state", "asking about current search state")

    # ── Priority 5: Capabilities / identity question ──────────────────────────
    # Only fire if the message does NOT also contain a shopping/product request.
    # "hi, recommend me white shoes" should go to new_search, not ask_capabilities.
    if _CAPABILITIES_RE.search(text) and not _RECOMMEND_RE.search(text):
        return PlannerDecision("ask_capabilities", "asking about assistant capabilities")

    # ── Priority 6: Explain match quality (needs active context) ─────────────
    if has_active_products and _EXPLAIN_QUALITY_RE.search(text):
        return PlannerDecision("explain_match_quality", "asking why results are approximate")

    # ── Priority 7: Next result (short phrase + active pool) ──────────────────
    if has_active_products and _NEXT_RESULT_RE.search(text) and len(text.split()) <= 8:
        return PlannerDecision("next_result", "requesting next item from accepted pool")

    # ── Priority 8: Explicit comparison ───────────────────────────────────────
    if has_active_products and _COMPARE_RE.search(text):
        return PlannerDecision("compare_results", "explicit comparison between current products")

    # ── Priority 8.5: Bundle / combo request ──────────────────────────────────
    # Comes before refine/followup so "hoodie and sneaker combo" routes here, not new_search.
    if _BUNDLE_RE.search(text):
        return PlannerDecision("bundle_search", "2-item bundle/combo request")

    # ── Priority 8.7: "Best one" / "which one" followup ───────────────────────
    # Must come before _NEW_SEARCH_SIGNAL_RE guard so "which would you recommend?" routes here
    # even though "recommend" appears in the new-search signal list.
    if has_active_products and _BEST_ONE_RE.search(text):
        return PlannerDecision("followup_on_results", "which-one or best-one question over current results")

    # ── Priority 9: Refine current search ────────────────────────────────────
    if has_active_products and _REFINE_RE.search(text) and not _NEW_SEARCH_SIGNAL_RE.search(text):
        return PlannerDecision("refine_search", "adding/changing constraints on current search")

    # ── Priority 10: Follow-up on results ────────────────────────────────────
    if has_active_products and _FOLLOWUP_RE.search(text) and not _NEW_SEARCH_SIGNAL_RE.search(text):
        return PlannerDecision("followup_on_results", "follow-up question on current results")

    # ── Ambiguous cases: use LLM with transcript for better context ───────────
    if api_key and has_active_products and len(text.split()) <= 8:
        try:
            return _plan_with_llm(
                text, search_state, has_active_products, api_key,
                chat_history=chat_history,
            )
        except Exception:
            pass

    # ── Default: new search ───────────────────────────────────────────────────
    return PlannerDecision("new_search", "product query, defaulting to new search")


_PLANNER_SYSTEM = """\
You are the planner for a fashion shopping assistant.
Given a user message, session context, and recent conversation, choose exactly ONE action.

Actions:
- ask_capabilities: user asking what the assistant can do or what it is
- ask_current_state: user asking about their current search or filters
- new_search: completely fresh search with new product criteria
- refine_search: narrow or add constraints to the CURRENT search (keep category, add color/price/style)
- followup_on_results: question about currently shown products (price, describe, sort)
- compare_results: explicit comparison between shown products
- explain_match_quality: user asking why results are approximate or close alternatives
- next_result: user wants one more item from the same search ("show me another")
- reset_context: start over completely
- out_of_scope: clearly not about fashion or shopping

Use the recent conversation transcript to infer context (follow-ups, pronouns, references).

Output ONLY valid JSON, no markdown:
{"action": "<action>", "reasoning": "<1 sentence>"}
"""


def _plan_with_llm(
    text: str,
    search_state: dict,
    has_active_products: bool,
    api_key: str,
    chat_history: Optional[list] = None,
) -> PlannerDecision:
    import anthropic
    client = anthropic.Anthropic(api_key=api_key)

    state_parts = []
    for k in ["category", "color", "max_price", "usage", "gender"]:
        if search_state.get(k):
            state_parts.append(f"{k}={search_state[k]}")
    state_str = ", ".join(state_parts) if state_parts else "none"

    # Include recent transcript so the model can reason over context
    transcript_str = ""
    if chat_history:
        lines = []
        for m in (chat_history or [])[-4:]:
            role = str(m.get("role", "user")).capitalize()
            content = str(m.get("content", ""))[:200]
            lines.append(f"{role}: {content}")
        if lines:
            transcript_str = "\nRecent conversation:\n" + "\n".join(lines)

    user_content = (
        f'Current user message: "{text}"\n'
        f"Active search filters: {state_str}\n"
        f"Has active product results on screen: {has_active_products}"
        + transcript_str
    )

    resp = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=128,
        system=_PLANNER_SYSTEM,
        messages=[{"role": "user", "content": user_content}],
    )
    raw = resp.content[0].text.strip()
    raw = re.sub(r"```[a-z]*\n?", "", raw).strip()
    data = json.loads(raw)
    action = data.get("action", "new_search")
    # Validate action is known
    _VALID_ACTIONS = {
        "ask_capabilities", "ask_current_state", "new_search", "refine_search",
        "followup_on_results", "compare_results", "explain_match_quality",
        "next_result", "reset_context", "out_of_scope", "image_search",
        "bundle_search",
    }
    if action not in _VALID_ACTIONS:
        action = "new_search"
    return PlannerDecision(action=action, reasoning=data.get("reasoning", ""))
