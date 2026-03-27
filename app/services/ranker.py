"""
Scores and re-ranks candidate products against parsed constraints.
Purely rule-based; no LLM.
"""

from app.services.query_parser import ParsedQuery
from app.services.retriever import RetrievalResult

# Additive score weights
_W = {
    "tfidf":    3.0,
    "category": 3.0,
    "color":    2.0,
    "usage":    2.0,
    "gender":   1.0,
    "season":   1.0,
    "price":    2.0,
}


def rank(
    candidates: list[RetrievalResult],
    constraints: ParsedQuery,
    top_k: int = 3,
) -> list[RetrievalResult]:
    """Return up to top_k products sorted by weighted score."""
    scored: list[tuple[float, RetrievalResult]] = []

    for p in candidates:
        score = p.tfidf_score * _W["tfidf"]

        if constraints.category and p.category == constraints.category:
            score += _W["category"]

        if constraints.color and p.base_color:
            if constraints.color.lower() in p.base_color.lower():
                score += _W["color"]

        if constraints.usage and p.usage:
            if constraints.usage.lower() == p.usage.lower():
                score += _W["usage"]

        if constraints.gender:
            norm = constraints.gender.capitalize()
            if p.gender_normalized in (norm, "Unisex"):
                score += _W["gender"]

        if constraints.season and p.season:
            if constraints.season.lower() == p.season.lower():
                score += _W["season"]

        if constraints.max_price and p.price <= constraints.max_price:
            score += _W["price"]

        scored.append((score, p))

    scored.sort(key=lambda x: -x[0])
    return [p for _, p in scored[:top_k]]
