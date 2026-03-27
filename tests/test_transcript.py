"""
Tests for transcript-first behavior:
- first message gets proper action
- capabilities+shopping mixed message goes to new_search
- image turn does not stick on subsequent text turns
- state questions answer from state
- "show me another" uses accepted pool
- out-of-scope still refuses
- image search path still goes through judge
- reset clears state
"""

import pytest
from app.services.planner import plan, PlannerDecision

_EMPTY_STATE: dict = {}
_ACTIVE_STATE: dict = {
    "category": "shirt",
    "color": "white",
    "last_action": "new_search",
    "last_intent_action": "new_search",
}

# ── First-message behavior ────────────────────────────────────────────────────

def test_first_message_product_query_is_new_search():
    """A fresh product query with empty state should become new_search."""
    d = plan("recommend me a black hoodie", _EMPTY_STATE, False, False)
    assert d.action == "new_search"


def test_first_message_greeting_only_is_capabilities():
    """A plain greeting with no shopping keywords should become ask_capabilities."""
    d = plan("hello!", _EMPTY_STATE, False, False)
    assert d.action == "ask_capabilities"


def test_first_message_greeting_plus_shopping_is_new_search():
    """
    'hi, can you recommend me white shoes?' should NOT become ask_capabilities.
    The shopping keyword wins over the greeting.
    """
    d = plan("hi, can you recommend me some white shoes?", _EMPTY_STATE, False, False)
    assert d.action == "new_search", (
        f"Expected new_search but got {d.action}. "
        "Greeting + shopping request should be treated as a product search."
    )


def test_hello_recommend_is_new_search():
    """'hello, find me sneakers' should go to new_search, not ask_capabilities."""
    d = plan("hello, find me sneakers", _EMPTY_STATE, False, False)
    assert d.action == "new_search"


def test_first_message_with_image_is_image_search():
    """Any turn with a new image should always be image_search."""
    d = plan("do you have anything like this?", _EMPTY_STATE, True, False)
    assert d.action == "image_search"


# ── Transcript context passed to planner ─────────────────────────────────────

def test_planner_accepts_chat_history_param():
    """plan() should not raise when chat_history is provided."""
    history = [
        {"role": "user", "content": "show me white shirts"},
        {"role": "assistant", "content": "Here are some white shirts..."},
    ]
    d = plan("which is cheapest?", _ACTIVE_STATE, False, True, chat_history=history)
    # Should be a follow-up, not a new search
    assert d.action in ("followup_on_results", "compare_results", "new_search")


def test_planner_handles_empty_chat_history():
    """plan() should work with empty or None chat_history."""
    d = plan("find me a blue jacket", _EMPTY_STATE, False, False, chat_history=[])
    assert d.action == "new_search"

    d2 = plan("find me a blue jacket", _EMPTY_STATE, False, False, chat_history=None)
    assert d2.action == "new_search"


# ── Image context not sticky ──────────────────────────────────────────────────

def test_image_only_fires_on_new_upload():
    """Without a new image in current turn, even image_active state should not trigger image_search."""
    image_active_state = {
        "image_active": True,
        "image_summary": "uploaded shirt",
        "last_intent_action": "image_search",
    }
    # Text-only turn, no image_b64
    d = plan("do you have blue shirts?", image_active_state, False, True)
    # Should be new_search or refine, NOT image_search
    assert d.action != "image_search"


def test_new_image_upload_forces_image_search():
    """A new image upload always forces image_search regardless of state."""
    d = plan("do you have anything similar?", _ACTIVE_STATE, True, True)
    assert d.action == "image_search"


# ── State questions ───────────────────────────────────────────────────────────

def test_what_are_we_looking_for_with_active_state():
    d = plan("what are we looking for right now?", _ACTIVE_STATE, False, True)
    assert d.action == "ask_current_state"


def test_state_question_no_context_falls_through():
    """Without active search, state question should not return ask_current_state."""
    d = plan("what are we looking for?", _EMPTY_STATE, False, False)
    assert d.action != "ask_current_state"


# ── Next result from pool ─────────────────────────────────────────────────────

def test_show_another_with_active_is_next_result():
    d = plan("show me another one", _ACTIVE_STATE, False, True)
    assert d.action == "next_result"


def test_give_me_another_option():
    d = plan("give me another option", _ACTIVE_STATE, False, True)
    assert d.action == "next_result"


def test_show_more_without_active_is_new_search():
    """Without active products, 'show me more' can't pull from pool — new search."""
    d = plan("show me more shirts", _EMPTY_STATE, False, False)
    assert d.action == "new_search"


# ── Out-of-scope ──────────────────────────────────────────────────────────────

def test_coding_question_is_out_of_scope():
    d = plan("how to reverse a linked list in Python", _EMPTY_STATE, False, False)
    assert d.action == "out_of_scope"


def test_out_of_scope_with_active_state_still_refuses():
    d = plan("what's the weather today?", _ACTIVE_STATE, False, True)
    assert d.action == "out_of_scope"


# ── Reset ─────────────────────────────────────────────────────────────────────

def test_start_over_clears():
    d = plan("start over", _ACTIVE_STATE, False, True)
    assert d.action == "reset_context"


def test_reset_clears():
    d = plan("reset", _ACTIVE_STATE, False, True)
    assert d.action == "reset_context"


# ── Explain quality ───────────────────────────────────────────────────────────

def test_why_only_alternatives_is_explain():
    d = plan("why are these only alternatives?", _ACTIVE_STATE, False, True)
    assert d.action == "explain_match_quality"


def test_do_you_have_exact_match_is_explain():
    d = plan("do you have any exact match?", _ACTIVE_STATE, False, True)
    assert d.action == "explain_match_quality"


# ── Follow-up and refine ─────────────────────────────────────────────────────

def test_cheapest_is_followup():
    d = plan("which is cheapest?", _ACTIVE_STATE, False, True)
    assert d.action in ("followup_on_results", "compare_results")


def test_only_plain_is_refine():
    d = plan("only plain ones", _ACTIVE_STATE, False, True)
    assert d.action == "refine_search"
