"""
Tests for the orchestrator layer.

Covers:
- ToolBundle construction
- _build_orchestrator_context: user message, action, judge summary, products, transcript, constraints
- compose_turn: raises LLMUnavailableError on no API key (no fallback, no products)
"""

import pytest
from app.services.orchestrator import ToolBundle, LLMUnavailableError, _build_orchestrator_context, compose_turn
from app.services.judge import JudgeResult, CandidateJudgment
from app.schemas import ProductItem


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_product(name: str = "Test Product", category: str = "Hoodie", price: float = 50.0) -> ProductItem:
    return ProductItem(
        id=1,
        product_name=name,
        category=category,
        base_color="white",
        usage="Casual",
        season="Summer",
        gender="Unisex",
        price=price,
        image_path="images/1.jpg",
        reason="Test reason",
    )


def _make_judge(exact: int = 0, close: int = 0, summary: str = "") -> JudgeResult:
    judgments = []
    for i in range(exact):
        judgments.append(CandidateJudgment(product_id=i, verdict="exact_match", reason=""))
    for i in range(close):
        judgments.append(CandidateJudgment(product_id=exact + i, verdict="close_alternative", reason=""))
    return JudgeResult(judgments=judgments, overall_summary=summary)


def _bundle(
    user_message: str = "show me white hoodies",
    action: str = "new_search",
    n_exact: int = 0,
    n_close: int = 0,
    judge_summary: str = "",
    filter_context: str = "",
    shown: list = None,
    state_summary: str = "(no active filters)",
    chat_history: list = None,
    image_dominant_color: str = "",
) -> ToolBundle:
    return ToolBundle(
        user_message=user_message,
        action=action,
        chat_history=chat_history or [],
        search_state_summary=state_summary,
        active_products=[],
        shown_products=shown or [],
        n_exact=n_exact,
        n_close=n_close,
        judge_result=_make_judge(n_exact, n_close, judge_summary) if (n_exact or n_close or judge_summary) else None,
        filter_context=filter_context,
        image_dominant_color=image_dominant_color,
    )


def _bundle_with_constraints(
    user_message: str,
    n_exact: int = 0,
    n_close: int = 0,
    requested_count: int = None,
    brand: str = None,
    min_price: float = None,
    max_price: float = None,
    judge_summary: str = "",
    shown: list = None,
) -> ToolBundle:
    return ToolBundle(
        user_message=user_message,
        action="new_search",
        chat_history=[],
        search_state_summary="(no active filters)",
        active_products=[],
        shown_products=shown or [],
        n_exact=n_exact,
        n_close=n_close,
        judge_result=_make_judge(n_exact, n_close, judge_summary) if (n_exact or n_close or judge_summary) else None,
        requested_count=requested_count,
        brand=brand,
        min_price=min_price,
        max_price=max_price,
    )


# ── ToolBundle construction ────────────────────────────────────────────────────

def test_tool_bundle_fields_set_correctly():
    b = _bundle(user_message="find red shoes", action="new_search", n_exact=2)
    assert b.user_message == "find red shoes"
    assert b.action == "new_search"
    assert b.n_exact == 2
    assert b.n_close == 0


def test_tool_bundle_shown_products_list():
    p = _make_product("Red Sneaker", "Sneakers", 60.0)
    b = _bundle(shown=[p])
    assert len(b.shown_products) == 1
    assert b.shown_products[0].product_name == "Red Sneaker"


# ── Context builder ────────────────────────────────────────────────────────────

def test_context_builder_includes_user_message():
    b = _bundle(user_message="find a blue jacket")
    ctx = _build_orchestrator_context(b)
    assert "find a blue jacket" in ctx


def test_context_builder_includes_action():
    b = _bundle(action="refine_search")
    ctx = _build_orchestrator_context(b)
    assert "Refined search" in ctx


def test_context_builder_includes_judge_summary():
    b = _bundle(judge_summary="Only grey jackets found, no blue.")
    ctx = _build_orchestrator_context(b)
    assert "Only grey jackets" in ctx


def test_context_builder_includes_shown_products():
    p = _make_product("Blue Jacket", "Jackets", 80.0)
    b = _bundle(shown=[p])
    ctx = _build_orchestrator_context(b)
    assert "Blue Jacket" in ctx


def test_context_builder_includes_transcript():
    history = [
        {"role": "user", "content": "show me hoodies"},
        {"role": "assistant", "content": "Here are some hoodies."},
    ]
    b = _bundle(chat_history=history)
    ctx = _build_orchestrator_context(b)
    assert "show me hoodies" in ctx


def test_context_builder_trims_transcript_to_10():
    """Only last 10 turns should be included."""
    history = [{"role": "user", "content": f"turn {i}"} for i in range(20)]
    b = _bundle(chat_history=history)
    ctx = _build_orchestrator_context(b)
    assert "turn 19" in ctx     # last turn present
    assert "turn 0" not in ctx  # oldest turn trimmed


def test_context_builder_image_search_includes_color():
    b = _bundle(action="image_search", image_dominant_color="navy")
    ctx = _build_orchestrator_context(b)
    assert "navy" in ctx


def test_context_builder_filter_context_included():
    b = _bundle(filter_context="User asked for blue but none match.")
    ctx = _build_orchestrator_context(b)
    assert "User asked for blue" in ctx


def test_context_builder_shows_constraints_block():
    """Constraints block appears in context when min_price/brand/count are set."""
    b = _bundle_with_constraints(
        "three Lee Cooper shoes over 100",
        requested_count=3,
        brand="Lee Cooper",
        min_price=100.0,
    )
    ctx = _build_orchestrator_context(b)
    assert "Lee Cooper" in ctx
    assert "100" in ctx
    assert "3" in ctx


def test_context_builder_no_constraints_block_when_empty():
    """No constraints block when none are specified."""
    b = _bundle(user_message="casual hoodie")
    ctx = _build_orchestrator_context(b)
    assert "Search constraints:" not in ctx


# ── compose_turn: no API key → raises LLMUnavailableError ─────────────────────

def test_compose_turn_no_api_key_raises():
    b = _bundle(user_message="find sneakers", n_close=2)
    with pytest.raises(LLMUnavailableError):
        compose_turn(b, api_key="")


def test_compose_turn_no_api_key_raises_for_exact_match():
    p = _make_product("White Hoodie", "Hoodie", 45.0)
    b = _bundle(user_message="white hoodie", n_exact=1, shown=[p])
    with pytest.raises(LLMUnavailableError):
        compose_turn(b, api_key="")


def test_compose_turn_no_api_key_raises_for_image_search():
    b = _bundle(
        user_message="find something like this",
        action="image_search",
        state_summary="image search, dominant color: red",
        image_dominant_color="red",
    )
    with pytest.raises(LLMUnavailableError):
        compose_turn(b, api_key="")


def test_compose_turn_no_api_key_raises_for_zero_results():
    b = _bundle_with_constraints("shoes over 100", min_price=100.0, n_exact=0, n_close=0)
    with pytest.raises(LLMUnavailableError):
        compose_turn(b, api_key="")
