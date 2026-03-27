"""
Image search: encodes a query image with CLIP, then finds nearest catalog products
by cosine similarity against precomputed embeddings (.npy).

After CLIP retrieval, a metadata + visual-heuristic reranker further sorts results
to penalise obvious mismatches (wrong colour family, heavy patterns vs plain query).
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from app.services.retriever import RetrievalResult, _row_to_result


@dataclass
class ImageSearchIndex:
    embeddings: np.ndarray     # [N, 512] float32, L2-normalized
    ids: np.ndarray            # [N] int64
    id_to_row: dict[int, pd.Series]  # product id -> catalog row


def build_image_index(
    embeddings: np.ndarray,
    embedding_ids: np.ndarray,
    df: pd.DataFrame,
) -> ImageSearchIndex:
    """Build the in-memory image search index from precomputed .npy data."""
    id_to_row = {int(row["id"]): row for _, row in df.iterrows()}
    return ImageSearchIndex(
        embeddings=embeddings,
        ids=embedding_ids,
        id_to_row=id_to_row,
    )


def search_by_image(
    query_image,  # PIL.Image.Image
    index: ImageSearchIndex,
    processor,    # CLIPProcessor
    model,        # CLIPModel
    top_k: int = 20,
) -> list[RetrievalResult]:
    """
    Encode query_image with CLIP and return the top_k most similar catalog products.
    processor and model are passed in (loaded once at startup).
    """
    import torch

    device = next(model.parameters()).device
    inputs = processor(images=[query_image], return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        vec = model.get_image_features(**inputs)
        vec = vec / vec.norm(dim=-1, keepdim=True)  # L2-normalize

    query_vec = vec.cpu().numpy().astype(np.float32)  # [1, 512]

    # Cosine similarity: since both sides are L2-normalized, dot product == cosine
    scores = (index.embeddings @ query_vec.T).flatten()  # [N]

    top_local = np.argsort(-scores)[:top_k]

    results = []
    for i in top_local:
        pid = int(index.ids[i])
        row = index.id_to_row.get(pid)
        if row is None:
            continue
        result = _row_to_result(row)
        result.tfidf_score = float(scores[i])  # reused as visual similarity score
        results.append(result)
    return results


# ── Visual heuristics ──────────────────────────────────────────────────────────

_LIGHT_COLORS = {"white", "beige", "cream", "ivory", "off white", "off-white"}
_DARK_COLORS  = {"black", "navy", "charcoal", "dark grey", "dark gray", "dark blue"}
_WARM_COLORS  = {"red", "orange", "pink", "burgundy", "maroon", "coral", "peach"}
_COOL_COLORS  = {"blue", "green", "teal", "purple", "violet", "indigo", "cyan"}
_GREY_COLORS  = {"grey", "gray", "silver", "light grey", "light gray"}


def _color_family(color: str) -> Optional[str]:
    if not color:
        return None
    c = color.lower().strip()
    if c in _LIGHT_COLORS or "white" in c:
        return "light"
    if c in _DARK_COLORS or "black" in c or "dark" in c:
        return "dark"
    if c in _WARM_COLORS:
        return "warm"
    if c in _COOL_COLORS:
        return "cool"
    if c in _GREY_COLORS or "grey" in c or "gray" in c:
        return "grey"
    return "other"


def estimate_dominant_color(img) -> Optional[str]:
    """
    Return an approximate colour name for the dominant hue of the image.
    Uses simple pixel-average statistics on a down-sampled copy.
    """
    img_small = img.resize((50, 50))
    arr = np.array(img_small).astype(float)
    if arr.ndim < 3 or arr.shape[2] < 3:
        return None
    r_mean = arr[:, :, 0].mean()
    g_mean = arr[:, :, 1].mean()
    b_mean = arr[:, :, 2].mean()
    brightness = (r_mean + g_mean + b_mean) / 3

    if brightness > 210:
        return "white"
    if brightness < 45:
        return "black"
    # Grey: channels are close to each other and mid-brightness
    channel_range = max(r_mean, g_mean, b_mean) - min(r_mean, g_mean, b_mean)
    if channel_range < 25 and brightness > 100:
        return "grey"
    if channel_range < 25 and brightness <= 100:
        return "dark grey"
    # Chromatic colours
    if r_mean > g_mean + 25 and r_mean > b_mean + 25:
        return "red" if r_mean > 160 else "burgundy"
    if b_mean > r_mean + 20 and b_mean > g_mean + 20:
        return "blue" if b_mean > 130 else "navy"
    if g_mean > r_mean + 20 and g_mean > b_mean + 20:
        return "green"
    if r_mean > 180 and g_mean > 160 and b_mean < 100:
        return "yellow"
    if r_mean > 180 and g_mean < 100 and b_mean > 150:
        return "purple"
    if r_mean > 200 and g_mean > 120 and b_mean < 80:
        return "orange"
    if r_mean > 200 and g_mean < 140 and b_mean > 120:
        return "pink"
    return None


def estimate_visual_complexity(img) -> float:
    """
    Estimate how visually plain (0.0) vs patterned/complex (1.0) the image is.
    High local variance = stripes / patterns / prints.
    """
    img_small = img.resize((40, 40))
    arr = np.array(img_small).astype(float)
    if arr.ndim < 3:
        return 0.0
    # Variance of each pixel's channel values as proxy for local complexity
    variance = arr.var(axis=(0, 1)).mean()
    # Empirically: plain ~50-400, heavily patterned ~1500+
    return float(min(1.0, variance / 1200.0))


def rerank_image_results(
    results: list[RetrievalResult],
    query_image,
    text_constraints=None,   # app.services.query_parser.ParsedQuery or None
    top_k: int = 3,
) -> tuple[list[RetrievalResult], str]:
    """
    Re-score CLIP results using metadata + visual heuristics.

    Adjustments applied to the base CLIP cosine similarity score:
      +0.10  exact colour match (dominant vs product base_color)
      +0.05  same colour family
      −0.06  colour-family mismatch
      −0.05  product is "multi" when query image appears visually plain
      +0.08  category matches text constraint
      +0.06  colour matches text constraint

    Returns (reranked_top_k, dominant_color_or_empty_string).
    """
    dominant_color = estimate_dominant_color(query_image)
    is_plain = estimate_visual_complexity(query_image) < 0.35

    query_family = _color_family(dominant_color) if dominant_color else None

    scored: list[tuple[float, RetrievalResult]] = []
    for r in results:
        score = r.tfidf_score  # CLIP cosine similarity as base

        # Colour matching
        prod_color = (r.base_color or "").lower().strip()
        if dominant_color and prod_color:
            if dominant_color in prod_color or prod_color in dominant_color:
                score += 0.10   # strong match
            elif query_family and _color_family(prod_color) == query_family:
                score += 0.05   # same family (e.g. white → beige)
            else:
                score -= 0.06   # colour mismatch penalty

        # Plain-query pattern penalty
        if is_plain and "multi" in prod_color:
            score -= 0.05

        # Text constraint boosts
        if text_constraints:
            if text_constraints.category and r.category == text_constraints.category:
                score += 0.08
            if text_constraints.color and prod_color:
                if text_constraints.color.lower() in prod_color:
                    score += 0.06
            if text_constraints.usage and r.usage:
                if text_constraints.usage.lower() == r.usage.lower():
                    score += 0.03

        scored.append((score, r))

    scored.sort(key=lambda x: -x[0])
    return [r for _, r in scored[:top_k]], (dominant_color or "")


def load_clip_model(model_name: str = "openai/clip-vit-base-patch32"):
    """Load CLIP processor and model onto the best available device. Call once at startup."""
    import torch
    from transformers import CLIPModel, CLIPProcessor
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = CLIPProcessor.from_pretrained(model_name)
    model = CLIPModel.from_pretrained(model_name)
    model.eval()
    model = model.to(device)
    print(f"  CLIP model loaded on {device}.", flush=True)
    return processor, model
