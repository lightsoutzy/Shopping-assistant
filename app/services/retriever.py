"""
Text retriever: TF-IDF over searchable_text with optional metadata pre-filtering.
The TfidfRetriever is built once from the catalog DataFrame and reused per request.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class RetrievalResult:
    id: int
    product_name: str
    category: str
    brand: str | None
    base_color: str | None
    usage: str | None
    season: str | None
    gender: str
    gender_normalized: str
    price: float
    image_path: str
    description: str
    tfidf_score: float = 0.0


class TfidfRetriever:
    def __init__(self, df: pd.DataFrame) -> None:
        self._df = df.reset_index(drop=True)
        self._vectorizer = TfidfVectorizer(
            strip_accents="unicode",
            lowercase=True,
            ngram_range=(1, 2),
            min_df=1,
            sublinear_tf=True,
        )
        self._matrix = self._vectorizer.fit_transform(self._df["searchable_text"].fillna(""))

    # ── public API ────────────────────────────────────────────────────────────

    def search(
        self,
        query: str = "",
        *,
        category: str | None = None,
        brand: str | None = None,
        color: str | None = None,
        gender: str | None = None,
        usage: str | None = None,
        max_price: float | None = None,
        min_price: float | None = None,
        top_k: int = 20,
    ) -> list[RetrievalResult]:
        """
        Filter catalog by hard constraints, then rank remaining rows by
        TF-IDF cosine similarity against query. Returns up to top_k results.
        Falls back to unfiltered search if hard filtering leaves < 5 rows.
        """
        mask = self._build_mask(
            category=category,
            brand=brand,
            color=color,
            gender=gender,
            usage=usage,
            max_price=max_price,
            min_price=min_price,
        )
        candidate_idx = np.where(mask)[0]

        # Relax filters if too few candidates
        if len(candidate_idx) < 5:
            candidate_idx = np.arange(len(self._df))

        if query.strip():
            scores = self._score(query, candidate_idx)
        else:
            # No query text: return top_k by index order (stable / arbitrary)
            scores = np.zeros(len(candidate_idx))

        top_local = np.argsort(-scores)[:top_k]
        top_idx = candidate_idx[top_local]

        results = []
        for rank, idx in enumerate(top_idx):
            row = self._df.iloc[idx]
            results.append(_row_to_result(row, tfidf_score=float(scores[rank])))
        return results

    # ── internals ─────────────────────────────────────────────────────────────

    def _build_mask(
        self,
        category: str | None,
        brand: str | None,
        color: str | None,
        gender: str | None,
        usage: str | None,
        max_price: float | None,
        min_price: float | None = None,
    ) -> np.ndarray:
        mask = np.ones(len(self._df), dtype=bool)
        df = self._df

        if category:
            mask &= df["category"].str.lower() == category.lower()
        if brand:
            mask &= df["brand"].str.lower().eq(brand.lower()).fillna(False)
        if color:
            mask &= df["base_color"].str.lower().str.contains(color.lower(), na=False)
        if gender:
            norm = gender.capitalize()
            # Unisex matches any gender filter
            mask &= (df["gender_normalized"] == norm) | (df["gender_normalized"] == "Unisex")
        if usage:
            mask &= df["usage"].str.lower().eq(usage.lower()).fillna(False)
        if max_price is not None:
            mask &= df["price"] <= max_price
        if min_price is not None:
            mask &= df["price"] >= min_price

        return mask

    def _score(self, query: str, candidate_idx: np.ndarray) -> np.ndarray:
        query_vec = self._vectorizer.transform([query])
        candidate_matrix = self._matrix[candidate_idx]
        scores = cosine_similarity(query_vec, candidate_matrix).flatten()
        return scores


# ── helper ────────────────────────────────────────────────────────────────────

def _row_to_result(row: pd.Series, tfidf_score: float = 0.0) -> RetrievalResult:
    return RetrievalResult(
        id=int(row["id"]),
        product_name=str(row["product_name"]),
        category=str(row["category"]),
        brand=row["brand"] if pd.notna(row.get("brand")) else None,
        base_color=row["base_color"] if pd.notna(row.get("base_color")) else None,
        usage=row["usage"] if pd.notna(row.get("usage")) else None,
        season=row["season"] if pd.notna(row.get("season")) else None,
        gender=str(row["gender"]),
        gender_normalized=str(row["gender_normalized"]),
        price=float(row["price"]),
        image_path=str(row["image_path"]),
        description=str(row["description"]),
        tfidf_score=tfidf_score,
    )
