from typing import Any, Optional

from pydantic import BaseModel, Field


class AcceptedProduct(BaseModel):
    """One entry in the accepted pool from the judge's last run."""
    product_id: int
    verdict: str   # "exact_match" | "close_alternative"
    reason: str


class SearchState(BaseModel):
    """Structured search state maintained across conversational turns."""
    category: Optional[str] = None
    color: Optional[str] = None
    max_price: Optional[float] = None
    min_price: Optional[float] = None  # "over $100"
    brand: Optional[str] = None        # specific brand requested
    gender: Optional[str] = None
    usage: Optional[str] = None        # Casual | Sports | Formal | …
    plain_only: bool = False           # user wants plain/solid items
    # Metadata about the last retrieval
    last_action: str = ""
    last_query_summary: str = ""       # human-readable summary of active filters
    last_intent_action: str = ""       # planner action from last turn
    result_count: int = 0
    had_perfect_matches: Optional[bool] = None
    rejected_summary: str = ""        # why candidates were rejected (for explain_match_quality)
    # Pool for "show me another one" flow
    accepted_pool: list[AcceptedProduct] = Field(default_factory=list)
    shown_product_ids: list[int] = Field(default_factory=list)
    image_active: bool = False         # whether an image search is active context
    image_summary: str = ""           # description of uploaded image context


def _state_summary(state: "SearchState") -> str:
    """Return a concise human-readable description of active filters."""
    parts = []
    if state.category:
        parts.append(state.category)
    if state.color:
        parts.append(f"{state.color} color")
    if state.brand:
        parts.append(f"brand: {state.brand}")
    if state.min_price:
        parts.append(f"over ${state.min_price:.0f}")
    if state.max_price:
        parts.append(f"under ${state.max_price:.0f}")
    if state.gender:
        parts.append(f"for {state.gender}")
    if state.usage:
        parts.append(f"{state.usage} style")
    if state.plain_only:
        parts.append("plain/solid only")
    if state.image_active and state.image_summary:
        parts.append(f"[image: {state.image_summary}]")
    return ", ".join(parts) if parts else "(no active filters)"


class AgentRequest(BaseModel):
    text: str
    image_b64: Optional[str] = None        # base64-encoded JPEG/PNG; triggers image search
    session_id: Optional[str] = None       # used for per-session rate limiting
    # Backward-compat top-level overrides (used by existing tests / direct callers)
    category: Optional[str] = None
    max_price: Optional[float] = None
    # Conversational context
    chat_history: list[dict[str, Any]] = []
    active_products: list[dict[str, Any]] = []
    search_state: SearchState = Field(default_factory=SearchState)


class ProductItem(BaseModel):
    id: int
    product_name: str
    brand: Optional[str] = None
    category: str
    base_color: Optional[str] = None
    usage: Optional[str] = None
    season: Optional[str] = None
    gender: str
    price: float
    image_path: str    # relative to DATASET_ROOT; frontend resolves to a served URL
    reason: str        # one-sentence explanation
    match_tier: str = "close"   # "perfect" | "close"
    match_score: int = 0        # 0–100


class AgentResponse(BaseModel):
    intent: str        # planner action label
    message: str       # natural language reply
    products: list[ProductItem] = []
    bundle_pairs: list[list[ProductItem]] = Field(default_factory=list)   # for 2-item bundle results
    session_id: Optional[str] = None
    eval_passed: Optional[bool] = None
    context_action: str = "keep"    # "keep" | "replace" | "clear"
    search_state: SearchState = Field(default_factory=SearchState)
