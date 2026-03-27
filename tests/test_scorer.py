"""
Unit tests for the unified 0–100 candidate scorer.
"""

import pytest

from app.schemas import SearchState
from app.services.retriever import RetrievalResult
from app.services.scorer import (
    CLOSE_THRESHOLD,
    PERFECT_THRESHOLD,
    is_patterned,
    score_image_candidate,
    score_text_candidate,
    tier,
)


# ── helpers ───────────────────────────────────────────────────────────────────

def _make_product(
    *,
    category="t-shirt",
    base_color="white",
    price=40.0,
    usage="Casual",
    gender="Men",
    gender_normalized="Men",
    description="",
) -> RetrievalResult:
    return RetrievalResult(
        id=1,
        product_name="Test Product",
        category=category,
        brand=None,
        base_color=base_color,
        usage=usage,
        season="Summer",
        gender=gender,
        gender_normalized=gender_normalized,
        price=price,
        image_path="images/1.jpg",
        description=description,
        tfidf_score=0.3,
    )


def _state(**kwargs) -> SearchState:
    return SearchState(**kwargs)


# ── tier() ────────────────────────────────────────────────────────────────────

def test_tier_perfect():
    assert tier(80) == "perfect"
    assert tier(100) == "perfect"
    assert tier(95) == "perfect"


def test_tier_close():
    assert tier(60) == "close"
    assert tier(79) == "close"


def test_tier_reject():
    assert tier(0) == "reject"
    assert tier(59) == "reject"


# ── is_patterned() ────────────────────────────────────────────────────────────

def test_patterned_multi_color():
    p = _make_product(base_color="multicolor")
    assert is_patterned(p) is True


def test_patterned_description_stripe():
    p = _make_product(description="Classic striped cotton tee")
    assert is_patterned(p) is True


def test_patterned_floral():
    p = _make_product(description="Floral print summer dress")
    assert is_patterned(p) is True


def test_not_patterned_plain():
    p = _make_product(base_color="white", description="Plain cotton crewneck t-shirt")
    assert is_patterned(p) is False


# ── score_text_candidate() ────────────────────────────────────────────────────

def test_perfect_match_score():
    """Exact category + color + mid-range TF-IDF → should score ≥ 80."""
    p = _make_product(category="t-shirt", base_color="black", price=30.0, usage="Casual")
    s = _state(category="t-shirt", color="black", max_price=50.0, usage="Casual")
    score = score_text_candidate(p, s, tfidf_score=0.30)
    assert score >= PERFECT_THRESHOLD, f"Expected ≥{PERFECT_THRESHOLD}, got {score}"


def test_wrong_category_hard_reject():
    p = _make_product(category="shoes")
    s = _state(category="t-shirt")
    assert score_text_candidate(p, s, tfidf_score=0.5) == 0


def test_over_price_hard_reject():
    p = _make_product(price=120.0)
    s = _state(max_price=100.0)
    assert score_text_candidate(p, s, tfidf_score=0.5) == 0


def test_color_exact_match_no_deduction():
    p = _make_product(base_color="black")
    s = _state(color="black")
    score_no_color = score_text_candidate(p, _state(), tfidf_score=0.30)
    score_with_color = score_text_candidate(p, s, tfidf_score=0.30)
    # Exact color match should not deduct; no-color-constraint has an 8-pt "uncertainty" deduction
    assert score_with_color >= score_no_color


def test_wrong_color_family_penalty():
    p = _make_product(base_color="blue")
    s_same = _state(color="teal")     # both in _COOL_FAMILY → same family
    s_diff = _state(color="white")   # _LIGHT_FAMILY → different family

    score_same = score_text_candidate(p, s_same, tfidf_score=0.30)
    score_diff = score_text_candidate(p, s_diff, tfidf_score=0.30)
    assert score_same > score_diff, "Same color family should score better than different family"


def test_patterned_plain_only_penalty():
    p = _make_product(description="Bold striped pattern")
    s_no = _state()
    s_plain = _state(plain_only=True)
    score_no = score_text_candidate(p, s_no, tfidf_score=0.30)
    score_plain = score_text_candidate(p, s_plain, tfidf_score=0.30)
    assert score_plain < score_no


def test_low_tfidf_penalty():
    p = _make_product()
    s = _state()
    score_good = score_text_candidate(p, s, tfidf_score=0.30)
    score_poor = score_text_candidate(p, s, tfidf_score=0.02)
    assert score_good > score_poor


def test_score_clamped_0_to_100():
    p = _make_product()
    s = _state()
    for tfidf in [0.0, 0.01, 0.1, 0.5, 1.0]:
        score = score_text_candidate(p, s, tfidf_score=tfidf)
        assert 0 <= score <= 100, f"Score out of range: {score}"


# ── score_image_candidate() ───────────────────────────────────────────────────

def test_image_hard_reject_low_clip():
    p = _make_product()
    assert score_image_candidate(p, clip_similarity=0.64, dominant_color=None, is_plain_query=False) == 0


def test_image_excellent_clip_no_deduction():
    p = _make_product(base_color="white")
    score = score_image_candidate(p, clip_similarity=0.90, dominant_color="white", is_plain_query=False)
    assert score >= PERFECT_THRESHOLD


def test_image_color_mismatch_penalty():
    p = _make_product(base_color="black")
    score_match = score_image_candidate(p, 0.85, dominant_color="black", is_plain_query=False)
    score_mismatch = score_image_candidate(p, 0.85, dominant_color="white", is_plain_query=False)
    assert score_match > score_mismatch


def test_image_patterned_plain_query_penalty():
    p_plain = _make_product(base_color="white", description="Plain white t-shirt")
    p_pattern = _make_product(base_color="white", description="Striped white t-shirt")
    score_plain = score_image_candidate(p_plain, 0.85, dominant_color="white", is_plain_query=True)
    score_pattern = score_image_candidate(p_pattern, 0.85, dominant_color="white", is_plain_query=True)
    assert score_plain > score_pattern


def test_image_score_clamped():
    p = _make_product()
    for clip in [0.65, 0.70, 0.75, 0.80, 0.90, 1.0]:
        score = score_image_candidate(p, clip, dominant_color=None, is_plain_query=False)
        assert 0 <= score <= 100
