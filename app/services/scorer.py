"""
Unified candidate scorer (0–100) for both text and image retrieval.

Uses a "start at 100, deduct for mismatches" approach so that a product that
satisfies every hard constraint scores near 100, and obviously wrong matches
are pushed below the reject threshold.

Score tiers (PERFECT_THRESHOLD / CLOSE_THRESHOLD):
  80–100 : perfect match   – return up to 5
  60–79  : close alternative – return up to 5 with honest explanation
  0–59   : reject            – do not show
"""

from typing import Optional

from app.schemas import SearchState
from app.services.retriever import RetrievalResult

PERFECT_THRESHOLD = 80
CLOSE_THRESHOLD = 60

# ── Visual pattern detection ───────────────────────────────────────────────────

_PATTERN_WORDS = {
    "stripe", "striped", "stripes",
    "check", "checked", "plaid",
    "print", "printed", "graphic", "graphics",
    "floral", "pattern", "patterned",
    "geometric", "paisley", "camouflage", "camo",
    "abstract", "colorblock", "colour block", "color block",
    "tie-dye", "tie dye", "ombre",
}

# ── Colour families (for loose matching) ─────────────────────────────────────

_LIGHT_FAMILY = {"white", "beige", "cream", "ivory", "off white", "off-white", "ecru"}
_DARK_FAMILY  = {"black", "navy", "charcoal", "dark grey", "dark gray", "dark blue",
                 "dark green", "dark brown"}
_WARM_FAMILY  = {"red", "orange", "pink", "burgundy", "maroon", "coral", "peach",
                 "rust", "salmon", "rose"}
_COOL_FAMILY  = {"blue", "green", "teal", "purple", "violet", "indigo", "cyan",
                 "turquoise", "mint", "olive"}
_GREY_FAMILY  = {"grey", "gray", "silver", "light grey", "light gray", "dark grey",
                 "dark gray", "charcoal"}


def _color_family(color: str) -> Optional[str]:
    if not color:
        return None
    c = color.lower().strip()
    # Check membership in ordered priority (dark before light to avoid "dark grey" → light)
    if c in _DARK_FAMILY or "black" in c:
        return "dark"
    if c in _LIGHT_FAMILY or "white" in c:
        return "light"
    if c in _GREY_FAMILY or "grey" in c or "gray" in c:
        return "grey"
    if c in _WARM_FAMILY:
        return "warm"
    if c in _COOL_FAMILY:
        return "cool"
    return "other"


def is_patterned(product: RetrievalResult) -> bool:
    """Return True if the product appears visually patterned (not plain/solid)."""
    color = (product.base_color or "").lower()
    if "multi" in color:
        return True
    desc = (product.description or "").lower()
    return any(w in desc for w in _PATTERN_WORDS)


# ── Text-search scorer ────────────────────────────────────────────────────────

def score_text_candidate(
    product: RetrievalResult,
    state: SearchState,
    tfidf_score: float,
) -> int:
    """
    Score a text-retrieval candidate against the current SearchState.

    Starts at 100 and deducts for constraint mismatches.
    Hard constraints (category, price) immediately return 0 on violation.
    """
    score = 100

    # ── Category (hard constraint) ────────────────────────────────────────────
    if state.category:
        if product.category != state.category:
            return 0   # hard reject — wrong category
        # Perfect category match: no deduction
    else:
        # No category constraint: slight penalty since match is less certain
        score -= 8

    # ── Price (hard constraint) ───────────────────────────────────────────────
    if state.max_price is not None:
        if product.price > state.max_price:
            return 0   # hard reject — over budget
        # Within budget: small bonus for headroom
        if product.price <= state.max_price * 0.80:
            pass       # well within budget, no deduction
        else:
            score -= 5  # tight near the ceiling

    # ── Color match ───────────────────────────────────────────────────────────
    if state.color:
        prod_color = (product.base_color or "").lower()
        query_color = state.color.lower()
        if query_color in prod_color or prod_color in query_color:
            pass   # exact (or substring) match — no deduction
        elif _color_family(query_color) == _color_family(prod_color):
            score -= 15   # same family (e.g. white → beige)
        else:
            score -= 35   # wrong color family — significant penalty

    # ── Plain / pattern preference ────────────────────────────────────────────
    if state.plain_only:
        if is_patterned(product):
            score -= 30   # patterned item when user wants plain

    # ── Usage / style ─────────────────────────────────────────────────────────
    if state.usage and product.usage:
        if state.usage.lower() != product.usage.lower():
            score -= 10

    # ── Gender ────────────────────────────────────────────────────────────────
    if state.gender and product.gender_normalized:
        if product.gender_normalized not in (state.gender.capitalize(), "Unisex"):
            score -= 8

    # ── TF-IDF semantic relevance adjustment ─────────────────────────────────
    # Good TF-IDF ≥ 0.25, poor < 0.05
    if tfidf_score < 0.05:
        score -= 18
    elif tfidf_score < 0.15:
        score -= 10
    elif tfidf_score < 0.25:
        score -= 4

    return max(0, min(100, score))


# ── Image-search scorer ───────────────────────────────────────────────────────

def score_image_candidate(
    product: RetrievalResult,
    clip_similarity: float,       # raw cosine, typically 0.65–0.95
    dominant_color: Optional[str],
    is_plain_query: bool,
    text_state: Optional[SearchState] = None,
) -> int:
    """
    Score an image-search candidate.

    Starts at 100 and deducts for visual and metadata mismatches.
    CLIP similarity below 0.65 is a hard reject.
    """
    # Hard floor on CLIP similarity
    if clip_similarity < 0.65:
        return 0

    score = 100

    # ── CLIP similarity penalty ───────────────────────────────────────────────
    if clip_similarity >= 0.85:
        pass           # excellent — no deduction
    elif clip_similarity >= 0.80:
        score -= 5
    elif clip_similarity >= 0.75:
        score -= 15
    elif clip_similarity >= 0.70:
        score -= 25
    else:              # 0.65–0.70
        score -= 35

    prod_color = (product.base_color or "").lower()

    # ── Color match ───────────────────────────────────────────────────────────
    if dominant_color:
        dc = dominant_color.lower()
        if dc in prod_color or prod_color in dc:
            pass   # exact match
        elif _color_family(dc) == _color_family(prod_color):
            score -= 15   # same family
        else:
            score -= 35   # wrong color family

    # ── Plain vs patterned ────────────────────────────────────────────────────
    if is_plain_query:
        if is_patterned(product):
            score -= 30

    # ── Text constraint boosts ────────────────────────────────────────────────
    if text_state:
        if text_state.category:
            if product.category == text_state.category:
                pass   # matches
            else:
                score -= 20
        if text_state.color and prod_color:
            tc = text_state.color.lower()
            if tc not in prod_color and prod_color not in tc:
                score -= 10   # slight additional penalty if text also specified a color

    return max(0, min(100, score))


# ── Tiering helper ────────────────────────────────────────────────────────────

def tier(score: int) -> str:
    """Return 'perfect', 'close', or 'reject'."""
    if score >= PERFECT_THRESHOLD:
        return "perfect"
    if score >= CLOSE_THRESHOLD:
        return "close"
    return "reject"
