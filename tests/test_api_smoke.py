"""
Smoke tests for the /agent endpoint using FastAPI TestClient.
Requires data/processed/catalog.parquet to exist.

Without an ANTHROPIC_API_KEY, judge and orchestrator are unavailable, so product
searches return 0 products + LLM_UNAVAILABLE_MSG. Tests that require products
are skipped in that case.
"""

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_general_chat(client):
    r = client.post("/agent", json={"text": "hello"})
    assert r.status_code == 200
    body = r.json()
    assert body["intent"] == "ask_capabilities"
    assert len(body["message"]) > 0


def test_recommend_returns_products(client):
    r = client.post("/agent", json={"text": "find me a black t-shirt"})
    assert r.status_code == 200
    body = r.json()
    assert body["intent"] in ("new_search", "recommend_products")
    if not body.get("products"):
        pytest.skip("No products returned — requires ANTHROPIC_API_KEY")
    assert len(body["products"]) > 0


def test_recommend_respects_category(client):
    r = client.post("/agent", json={"text": "show me bags", "category": "bag"})
    assert r.status_code == 200
    body = r.json()
    if not body.get("products"):
        pytest.skip("No products returned — requires ANTHROPIC_API_KEY")
    for p in body["products"]:
        assert p["category"] == "bag"


def test_recommend_respects_price(client):
    r = client.post("/agent", json={"text": "jacket under 100", "max_price": 100.0})
    assert r.status_code == 200
    body = r.json()
    if not body.get("products"):
        pytest.skip("No products returned — requires ANTHROPIC_API_KEY")
    for p in body["products"]:
        assert p["price"] <= 100.0


def test_vague_input_returns_200(client):
    # "ok" has no shopping keywords — planner treats it as new_search (no match found)
    r = client.post("/agent", json={"text": "ok"})
    assert r.status_code == 200
    assert len(r.json()["message"]) > 0


def test_response_has_session_id(client):
    r = client.post("/agent", json={"text": "find me sneakers"})
    assert r.status_code == 200
    assert r.json()["session_id"] is not None


def test_eval_passed_for_valid_recommendation(client):
    r = client.post("/agent", json={"text": "blue casual shirt"})
    assert r.status_code == 200
    body = r.json()
    if body.get("eval_passed") is not None:
        assert body["eval_passed"] is True


# ── new: conversational route-level tests ─────────────────────────────────────

def test_out_of_scope_returns_no_products(client):
    """Out-of-scope prompt must return an empty product list and a refusal message."""
    r = client.post("/agent", json={"text": "how to reverse a linked list"})
    assert r.status_code == 200
    body = r.json()
    assert body["intent"] == "out_of_scope"
    assert body["products"] == []
    assert len(body["message"]) > 0
    # Refusal message should NOT look like a product recommendation
    assert "linked list" not in body["message"].lower() or "catalog" in body["message"].lower()


def test_out_of_scope_trivia_no_products(client):
    r = client.post("/agent", json={"text": "who invented the telephone"})
    assert r.status_code == 200
    body = r.json()
    assert body["intent"] == "out_of_scope"
    assert body["products"] == []


def test_context_action_replace_on_new_search(client):
    """A fresh product search returns context_action='replace' when products found, 'keep' on LLM error."""
    r = client.post("/agent", json={"text": "find me a black hoodie"})
    assert r.status_code == 200
    body = r.json()
    if body.get("products"):
        assert body.get("context_action") == "replace"
    # Without API key: LLM unavailable → context_action="keep", products=[]


def test_followup_color_filter_on_active_products(client):
    """
    Sending active_products with a colour follow-up should filter to that colour.
    Expects: only white products returned (or empty + honest message if none found).
    """
    active = [
        {"id": 1, "product_name": "White Crew", "brand": None,
         "category": "hoodie", "base_color": "white", "usage": "Casual",
         "season": "Winter", "gender": "Unisex", "price": 45.0,
         "image_path": "images/1.jpg", "reason": ""},
        {"id": 2, "product_name": "Grey Sweat", "brand": None,
         "category": "hoodie", "base_color": "grey", "usage": "Casual",
         "season": "Winter", "gender": "Unisex", "price": 50.0,
         "image_path": "images/2.jpg", "reason": ""},
    ]
    r = client.post("/agent", json={
        "text": "white ones?",
        "active_products": active,
    })
    assert r.status_code == 200
    body = r.json()
    assert body["intent"] == "followup_on_results"
    # Result must be the single white product, not the grey one
    if body["products"]:
        for p in body["products"]:
            assert "white" in (p.get("base_color") or "").lower()


def test_followup_no_match_returns_empty(client):
    """When follow-up filter finds nothing, backend must return empty products + honest message."""
    active = [
        {"id": 2, "product_name": "Grey Sweat", "brand": None,
         "category": "hoodie", "base_color": "grey", "usage": "Casual",
         "season": "Winter", "gender": "Unisex", "price": 50.0,
         "image_path": "images/2.jpg", "reason": ""},
    ]
    r = client.post("/agent", json={
        "text": "only red ones",
        "active_products": active,
    })
    assert r.status_code == 200
    body = r.json()
    assert body["intent"] == "followup_on_results"
    assert body["products"] == []
    # Message must signal no match OR LLM unavailable error
    msg = body["message"].lower()
    assert (
        "don't see" in msg or "no " in msg or "none" in msg
        or "couldn't find" in msg or "not found" in msg
        or "unavailable" in msg or "can't" in msg
    )


def test_different_batch_excludes_current_products(client):
    """'Show me a different batch' must return different products than active_products."""
    # First get some real products
    r1 = client.post("/agent", json={"text": "find me a casual hoodie"})
    assert r1.status_code == 200
    first_products = r1.json().get("products", [])

    if not first_products:
        pytest.skip("No products returned for initial search")

    first_ids = {p["id"] for p in first_products}

    # Now ask for a different batch
    r2 = client.post("/agent", json={
        "text": "show me a different batch",
        "active_products": first_products,
    })
    assert r2.status_code == 200
    second_products = r2.json().get("products", [])

    if second_products:
        second_ids = {p["id"] for p in second_products}
        # At least some products should be different
        assert second_ids != first_ids, "Expected different products for 'next batch' request"


# ── Constraint enforcement tests ──────────────────────────────────────────────

def test_min_price_products_are_above_threshold(client):
    """'Shoes over $80' must only return products with price >= 80."""
    r = client.post("/agent", json={"text": "shoes over 80"})
    assert r.status_code == 200
    body = r.json()
    for p in body.get("products", []):
        assert p["price"] >= 80.0, (
            f"Product {p['product_name']} at ${p['price']} violates min_price=80"
        )


def test_requested_count_limits_shown_products(client):
    """'Give me 2 shoes' must return at most 2 products."""
    r = client.post("/agent", json={"text": "give me 2 shoes"})
    assert r.status_code == 200
    body = r.json()
    assert len(body.get("products", [])) <= 2, (
        f"Expected at most 2 products for count request, got {len(body.get('products', []))}"
    )


def test_requested_count_three_returns_at_most_three(client):
    """'Show me three hoodies' must return at most 3 products."""
    r = client.post("/agent", json={"text": "show me three hoodies"})
    assert r.status_code == 200
    body = r.json()
    assert len(body.get("products", [])) <= 3, (
        f"Expected at most 3 products, got {len(body.get('products', []))}"
    )


def test_brand_query_returns_200(client):
    """A bare brand name query must return 200 and a message (not crash)."""
    r = client.post("/agent", json={"text": "Lee Cooper"})
    assert r.status_code == 200
    body = r.json()
    assert len(body["message"]) > 0


def test_brand_with_category_returns_200(client):
    """'Lee Cooper shoes' must return 200 and products or an honest no-match."""
    r = client.post("/agent", json={"text": "Lee Cooper shoes"})
    assert r.status_code == 200
    body = r.json()
    assert len(body["message"]) > 0
    # If products returned, they should all be Lee Cooper
    for p in body.get("products", []):
        assert (p.get("brand") or "").lower() == "lee cooper", (
            f"Expected Lee Cooper brand, got {p.get('brand')!r}"
        )


def test_min_price_message_does_not_say_couldnt_find_when_results_exist(client):
    """If min_price results exist, response must not say 'couldn't find exact match'."""
    r = client.post("/agent", json={"text": "shoes over 50"})
    assert r.status_code == 200
    body = r.json()
    if body.get("products"):
        msg = body["message"].lower()
        # Should NOT say exact match wasn't found when products ARE shown
        assert "i couldn't find an exact match" not in msg, (
            f"Found products but message says no exact match: {msg!r}"
        )
