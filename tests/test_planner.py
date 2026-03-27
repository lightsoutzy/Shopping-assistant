"""
Tests for the planner service.
No API calls — tests rule-based routing only.
"""

import pytest
from app.services.planner import plan, PlannerDecision

_EMPTY_STATE = {}
_ACTIVE_STATE = {"category": "shirt", "color": "white", "last_action": "new_search"}


# ── State questions ────────────────────────────────────────────────────────────

def test_state_question_with_active_search():
    d = plan("what's my current filter?", _ACTIVE_STATE, False, True)
    assert d.action == "ask_current_state"

def test_state_question_what_are_we_looking_for():
    d = plan("what are we looking for?", _ACTIVE_STATE, False, True)
    assert d.action == "ask_current_state"

def test_state_question_no_active_search_falls_through():
    # No active search → can't answer state question → falls to new_search / capabilities
    d = plan("what's my current filter?", _EMPTY_STATE, False, False)
    assert d.action != "ask_current_state"


# ── Capabilities ───────────────────────────────────────────────────────────────

def test_hello_is_capabilities():
    d = plan("hello", _EMPTY_STATE, False, False)
    assert d.action == "ask_capabilities"

def test_what_can_you_do():
    d = plan("what can you do?", _EMPTY_STATE, False, False)
    assert d.action == "ask_capabilities"

def test_who_are_you():
    d = plan("who are you?", _EMPTY_STATE, False, False)
    assert d.action == "ask_capabilities"


# ── Out-of-scope ───────────────────────────────────────────────────────────────

def test_linked_list_is_out_of_scope():
    d = plan("how to reverse a linked list", _EMPTY_STATE, False, False)
    assert d.action == "out_of_scope"

def test_weather_is_out_of_scope():
    d = plan("what's the weather today?", _EMPTY_STATE, False, False)
    assert d.action == "out_of_scope"

def test_recipe_is_out_of_scope():
    d = plan("give me a recipe for pasta", _EMPTY_STATE, False, False)
    assert d.action == "out_of_scope"


# ── Reset ─────────────────────────────────────────────────────────────────────

def test_start_over_is_reset():
    d = plan("start over", _ACTIVE_STATE, False, True)
    assert d.action == "reset_context"

def test_reset_is_reset():
    d = plan("reset", _ACTIVE_STATE, False, True)
    assert d.action == "reset_context"


# ── Image search ───────────────────────────────────────────────────────────────

def test_image_present_forces_image_search():
    d = plan("find something similar", _EMPTY_STATE, True, False)
    assert d.action == "image_search"

def test_image_overrides_everything():
    d = plan("start over", _ACTIVE_STATE, True, True)
    assert d.action == "image_search"


# ── Explain match quality ──────────────────────────────────────────────────────

def test_why_alternatives_with_active():
    d = plan("why are these only alternatives?", _ACTIVE_STATE, False, True)
    assert d.action == "explain_match_quality"

def test_do_you_have_exact_match_with_active():
    d = plan("do you have any exact match?", _ACTIVE_STATE, False, True)
    assert d.action == "explain_match_quality"

def test_explain_quality_no_active_is_not_triggered():
    # Without active products, explain_match_quality should not trigger
    d = plan("why are these only alternatives?", _EMPTY_STATE, False, False)
    assert d.action != "explain_match_quality"


# ── Next result ────────────────────────────────────────────────────────────────

def test_show_me_another_is_next_result():
    d = plan("show me another one", _ACTIVE_STATE, False, True)
    assert d.action == "next_result"

def test_give_me_another_option():
    d = plan("give me another option", _ACTIVE_STATE, False, True)
    assert d.action == "next_result"

def test_next_result_requires_active_products():
    d = plan("show me another one", _EMPTY_STATE, False, False)
    assert d.action != "next_result"


# ── New search ────────────────────────────────────────────────────────────────

def test_product_query_is_new_search():
    d = plan("find me a black casual jacket under $100", _EMPTY_STATE, False, False)
    assert d.action == "new_search"

def test_refine_plain_with_active():
    d = plan("only plain ones", _ACTIVE_STATE, False, True)
    assert d.action == "refine_search"


# ── Follow-up ─────────────────────────────────────────────────────────────────

def test_cheapest_with_active_is_followup():
    d = plan("which is cheapest?", _ACTIVE_STATE, False, True)
    assert d.action == "followup_on_results"

def test_compare_with_active():
    d = plan("compare these two", _ACTIVE_STATE, False, True)
    assert d.action == "compare_results"
