"""
Retriever tests — load the real catalog once per session.
Requires data/processed/catalog.parquet to exist.
"""

import pytest
import pandas as pd

from app.services.retriever import TfidfRetriever


@pytest.fixture(scope="module")
def retriever():
    df = pd.read_parquet("data/processed/catalog.parquet")
    return TfidfRetriever(df)


def test_free_text_returns_results(retriever):
    results = retriever.search("black sports jacket", top_k=5)
    assert len(results) > 0


def test_category_filter(retriever):
    results = retriever.search("casual", category="bag", top_k=10)
    assert all(r.category == "bag" for r in results)


def test_color_filter(retriever):
    results = retriever.search("shirt", color="black", top_k=10)
    assert all("black" in (r.base_color or "") for r in results)


def test_price_filter(retriever):
    results = retriever.search("jacket", max_price=80.0, top_k=10)
    assert all(r.price <= 80.0 for r in results)


def test_top_k_respected(retriever):
    results = retriever.search("shoes", top_k=3)
    assert len(results) <= 3


def test_constraint_relaxation_when_too_narrow(retriever):
    # Extremely restrictive filter that matches nothing → should still return results
    results = retriever.search(
        "shoe",
        category="shoes",
        color="ultraviolet",   # no such color in catalog
        top_k=5,
    )
    assert len(results) > 0


def test_gender_filter_includes_unisex(retriever):
    results = retriever.search("t-shirt", category="t-shirt", gender="Men", top_k=20)
    for r in results:
        assert r.gender_normalized in ("Men", "Unisex"), f"Unexpected gender: {r.gender_normalized}"
