"""Tests for the keyword-fallback parser (no API key required)."""

from app.services.query_parser import parse_with_keywords


def test_category_extracted():
    q = parse_with_keywords("show me a black jacket")
    assert q.category == "jacket"


def test_color_extracted():
    q = parse_with_keywords("I want a red t-shirt")
    assert q.color == "red"


def test_price_extracted():
    q = parse_with_keywords("sneakers under $80")
    assert q.max_price == 80.0


def test_price_extracted_no_dollar():
    q = parse_with_keywords("bag less than 50")
    assert q.max_price == 50.0


def test_gender_men():
    q = parse_with_keywords("men's casual shoes")
    assert q.gender == "Men"


def test_gender_women():
    q = parse_with_keywords("women's summer hoodie")
    assert q.gender == "Women"


def test_usage_sports():
    q = parse_with_keywords("sports t-shirt")
    assert q.usage == "Sports"


def test_season_winter():
    q = parse_with_keywords("winter jacket")
    assert q.season == "Winter"


def test_empty_query_returns_defaults():
    q = parse_with_keywords("")
    assert q.category is None
    assert q.color is None
    assert q.max_price is None
    assert q.free_text == ""


def test_alias_tshirt():
    q = parse_with_keywords("tshirt for gym")
    assert q.category == "t-shirt"


def test_alias_backpack():
    q = parse_with_keywords("a good backpack for travel")
    assert q.category == "bag"
    assert q.usage == "Travel"


# ── New constraint fields ──────────────────────────────────────────────────────

def test_min_price_over():
    q = parse_with_keywords("shoes over 100")
    assert q.min_price == 100.0


def test_min_price_above():
    q = parse_with_keywords("jacket above $80")
    assert q.min_price == 80.0


def test_min_price_more_than():
    q = parse_with_keywords("sneakers more than 60")
    assert q.min_price == 60.0


def test_min_price_at_least():
    q = parse_with_keywords("bag at least 50 dollars")
    assert q.min_price == 50.0


def test_max_and_min_price_together():
    """'between $50 and $100' — under extracts max, over not present in this form."""
    q = parse_with_keywords("hoodie under 100")
    assert q.max_price == 100.0
    assert q.min_price is None


def test_requested_count_digit():
    q = parse_with_keywords("show me 3 hoodies")
    assert q.requested_count == 3


def test_requested_count_word():
    q = parse_with_keywords("give me two sneakers")
    assert q.requested_count == 2


def test_requested_count_give_me_three():
    q = parse_with_keywords("give me three shoes over 100")
    assert q.requested_count == 3
    assert q.min_price == 100.0


def test_requested_count_find_five():
    q = parse_with_keywords("find me five jackets")
    assert q.requested_count == 5


def test_no_count_in_plain_query():
    q = parse_with_keywords("black casual sneakers")
    assert q.requested_count is None


def test_min_price_not_confused_with_max_price():
    q = parse_with_keywords("shoes over 100 under 200")
    assert q.min_price == 100.0
    assert q.max_price == 200.0
