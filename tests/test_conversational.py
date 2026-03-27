"""
Conversational behaviour tests: out-of-scope refusal, follow-up filtering,
no-match reporting, new-search detection, and image reranking heuristics.

These tests are unit-level and do NOT require the running FastAPI server,
the CLIP model, or the real product catalog.
"""

from unittest.mock import MagicMock

import numpy as np
import pytest

# ── intent routing ────────────────────────────────────────────────────────────

from app.services.intent_router import classify_with_context


class TestOutOfScope:
    def test_coding_question_refused(self):
        assert classify_with_context("how to reverse a linked list", False, []) == "out_of_scope"

    def test_algorithm_question_refused(self):
        assert classify_with_context("what is a binary search tree", False, []) == "out_of_scope"

    def test_coding_help_refused(self):
        assert classify_with_context("write me a function that sorts an array", False, []) == "out_of_scope"

    def test_trivia_refused(self):
        assert classify_with_context("who invented the telephone", False, []) == "out_of_scope"

    def test_geography_refused(self):
        assert classify_with_context("what is the capital of France", False, []) == "out_of_scope"

    def test_shopping_not_refused(self):
        assert classify_with_context("show me a black hoodie", False, []) != "out_of_scope"

    def test_general_chat_not_refused(self):
        assert classify_with_context("what can you do", False, []) != "out_of_scope"


# ── follow-up filter parsing ───────────────────────────────────────────────────

from app.api.routes_agent import _filter_active_products

_SAMPLE_PRODUCTS = [
    {
        "id": 1, "product_name": "Alpha White Crew", "brand": "Alpha",
        "category": "hoodie", "base_color": "white", "usage": "Casual",
        "season": "Summer", "gender": "Unisex", "price": 40.0,
        "image_path": "images/1.jpg", "reason": "",
    },
    {
        "id": 2, "product_name": "Benetton Black Sweat", "brand": "Benetton",
        "category": "hoodie", "base_color": "black", "usage": "Casual",
        "season": "Winter", "gender": "Unisex", "price": 60.0,
        "image_path": "images/2.jpg", "reason": "",
    },
    {
        "id": 3, "product_name": "SportX Grey Hoodie", "brand": "SportX",
        "category": "hoodie", "base_color": "grey", "usage": "Sports",
        "season": "Fall", "gender": "Men", "price": 50.0,
        "image_path": "images/3.jpg", "reason": "",
    },
]


class TestFollowupColorFilter:
    def test_white_filter_returns_white_only(self):
        filtered, desc, has_filter = _filter_active_products(_SAMPLE_PRODUCTS, "white ones?")
        assert has_filter
        assert all("white" in (p["base_color"] or "").lower() for p in filtered)
        assert len(filtered) == 1
        assert filtered[0]["id"] == 1

    def test_black_filter_returns_black_only(self):
        filtered, desc, has_filter = _filter_active_products(_SAMPLE_PRODUCTS, "only black ones")
        assert has_filter
        assert all("black" in (p["base_color"] or "").lower() for p in filtered)
        assert len(filtered) == 1

    def test_nonexistent_color_returns_empty(self):
        filtered, desc, has_filter = _filter_active_products(_SAMPLE_PRODUCTS, "show me the red ones")
        assert has_filter
        assert filtered == []


class TestFollowupBrandFilter:
    def test_benetton_filter(self):
        filtered, desc, has_filter = _filter_active_products(_SAMPLE_PRODUCTS, "only the Benetton one")
        assert has_filter
        assert len(filtered) == 1
        assert filtered[0]["brand"] == "Benetton"

    def test_unknown_brand_returns_empty(self):
        filtered, desc, has_filter = _filter_active_products(_SAMPLE_PRODUCTS, "only the Zara one")
        # "Zara" is not in the active products' brands so has_filter stays False
        # (we only filter on brands that appear in active results)
        assert not has_filter or filtered == []


class TestFollowupNoMatch:
    def test_no_match_flag_set(self):
        filtered, desc, has_filter = _filter_active_products(_SAMPLE_PRODUCTS, "only the red ones")
        assert has_filter
        assert filtered == []
        assert "red" in desc


class TestFollowupComparison:
    def test_cheapest_sort_no_hard_filter(self):
        """'Which is cheapest?' should sort but not set has_hard_filter."""
        filtered, desc, has_filter = _filter_active_products(_SAMPLE_PRODUCTS, "which is cheapest?")
        assert not has_filter          # comparison, not a colour/brand filter
        assert filtered[0]["price"] == min(p["price"] for p in _SAMPLE_PRODUCTS)

    def test_most_expensive_sort(self):
        filtered, desc, has_filter = _filter_active_products(_SAMPLE_PRODUCTS, "which is most expensive?")
        assert filtered[0]["price"] == max(p["price"] for p in _SAMPLE_PRODUCTS)


# ── new-search vs follow-up distinction ───────────────────────────────────────

class TestNewSearchVsFollowup:
    def test_actually_triggers_new_search(self):
        intent = classify_with_context(
            "actually show me something more sporty",
            False,
            _SAMPLE_PRODUCTS,
        )
        assert intent == "new_text_search"

    def test_instead_triggers_new_search(self):
        intent = classify_with_context(
            "instead show me summer dresses",
            False,
            _SAMPLE_PRODUCTS,
        )
        assert intent == "new_text_search"

    def test_which_is_cheapest_is_followup(self):
        intent = classify_with_context(
            "which is cheapest?",
            False,
            _SAMPLE_PRODUCTS,
        )
        assert intent == "followup_on_current_results"

    def test_only_benetton_is_followup(self):
        intent = classify_with_context(
            "only the Benetton one",
            False,
            _SAMPLE_PRODUCTS,
        )
        assert intent == "followup_on_current_results"


# ── image search reranking ────────────────────────────────────────────────────

from app.services.image_search import (
    estimate_dominant_color,
    estimate_visual_complexity,
    rerank_image_results,
)
from app.services.retriever import RetrievalResult


def _make_result(pid, color, tfidf=0.7, category="hoodie"):
    return RetrievalResult(
        id=pid, product_name=f"Product {pid}", category=category,
        brand=None, base_color=color, usage="Casual", season="Winter",
        gender="Unisex", gender_normalized="Unisex",
        price=50.0, image_path=f"images/{pid}.jpg",
        description="", tfidf_score=tfidf,
    )


class TestImageReranking:
    def _plain_white_image(self):
        """Return a solid white PIL image (plain, white dominant)."""
        from PIL import Image
        return Image.new("RGB", (100, 100), (240, 240, 240))

    def _dark_striped_image(self):
        """Return an image with alternating black/white stripes (patterned, dark)."""
        from PIL import Image
        import numpy as np
        arr = np.zeros((100, 100, 3), dtype=np.uint8)
        for row in range(100):
            val = 0 if (row // 10) % 2 == 0 else 255
            arr[row, :] = val
        return Image.fromarray(arr)

    def test_dominant_color_white_image(self):
        img = self._plain_white_image()
        color = estimate_dominant_color(img)
        assert color == "white"

    def test_visual_complexity_plain_is_low(self):
        img = self._plain_white_image()
        complexity = estimate_visual_complexity(img)
        assert complexity < 0.35, f"Expected plain image to have low complexity, got {complexity}"

    def test_visual_complexity_striped_is_high(self):
        img = self._dark_striped_image()
        complexity = estimate_visual_complexity(img)
        assert complexity >= 0.35, f"Expected striped image to have high complexity, got {complexity}"

    def test_white_query_boosts_white_products(self):
        img = self._plain_white_image()
        results = [
            _make_result(1, "white", tfidf=0.70),   # should score higher
            _make_result(2, "black", tfidf=0.72),   # slightly better CLIP but wrong color
            _make_result(3, "grey",  tfidf=0.68),
        ]
        reranked, dom_color = rerank_image_results(results, img, top_k=3)
        assert dom_color == "white"
        # White product (id=1) should beat the black product despite lower CLIP score
        ids = [r.id for r in reranked]
        assert ids.index(1) < ids.index(2), "White product should outrank black after reranking"

    def test_plain_query_penalises_multicolor(self):
        img = self._plain_white_image()
        results = [
            _make_result(1, "white",      tfidf=0.80),
            _make_result(2, "multi",      tfidf=0.82),  # multi-color pattern, higher CLIP
            _make_result(3, "off-white",  tfidf=0.75),
        ]
        reranked, _ = rerank_image_results(results, img, top_k=3)
        ids = [r.id for r in reranked]
        # Multi should be pushed below white/off-white despite higher raw CLIP score
        assert ids.index(1) < ids.index(2), "Plain query should penalise multi-color product"

    def test_image_search_followup_color_filter(self):
        """
        After an image search sets active_products, a follow-up 'white ones?'
        should filter those products by colour, not start a new search.
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
        intent = classify_with_context("white ones?", False, active)
        assert intent == "followup_on_current_results"

        filtered, desc, has_filter = _filter_active_products(active, "white ones?")
        assert has_filter
        assert len(filtered) == 1
        assert filtered[0]["id"] == 1
