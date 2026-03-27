"""
Preprocessing script: builds catalog.parquet from the raw Kaggle dataset.

Usage:
  python -m app.data.preprocess_dataset

After this, generate image embeddings with:
  python -m app.data.build_image_embeddings

See docs/preprocessing_plan.md for full step-by-step rationale.
"""

import json

import pandas as pd

from pathlib import Path

from app.config import (
    CATALOG_PATH,
    DATASET_ROOT,
)
from app.utils.text_utils import normalize_whitespace, strip_html

# ── Constants ─────────────────────────────────────────────────────────────────

CATEGORY_MAP: dict[str, str] = {
    "Tshirts": "t-shirt",
    "Shirts": "shirt",
    "Casual Shoes": "shoes",
    "Formal Shoes": "shoes",
    "Flats": "shoes",
    "Sports Shoes": "sneakers",
    "Jackets": "jacket",
    "Sweatshirts": "hoodie",
    "Sweaters": "hoodie",
    "Shorts": "shorts",
    "Handbags": "bag",
    "Backpacks": "bag",
    "Clutches": "bag",
}

CATEGORY_MULTIPLIERS: dict[str, float] = {
    "bag": 3.0,
    "jacket": 4.0,
    "shoes": 3.5,
    "sneakers": 3.0,
    "hoodie": 2.5,
    "shirt": 2.0,
    "t-shirt": 1.5,
    "shorts": 1.5,
}

# Use 9999 to mean "take all available"
SAMPLE_TARGETS: dict[str, int] = {
    "t-shirt": 250,
    "shirt": 200,
    "shoes": 200,
    "sneakers": 200,
    "jacket": 9999,
    "hoodie": 9999,
    "shorts": 200,
    "bag": 200,
}

RANDOM_SEED = 42


# ── Helpers ───────────────────────────────────────────────────────────────────

def mock_price(product_id: int, category: str) -> float:
    base = (product_id % 10) * 5 + 20
    return round(base * CATEGORY_MULTIPLIERS[category], 2)


def load_json_meta(styles_dir: Path, product_id: int) -> dict:
    """Load brand and description from per-product JSON. Silent on any failure."""
    path = styles_dir / f"{product_id}.json"
    if not path.exists():
        return {}
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            data = json.load(f).get("data", {})
        brand = data.get("brandName") or None
        raw_desc = (
            data.get("productDescriptors", {})
            .get("description", {})
            .get("value", "")
        )
        description = strip_html(raw_desc) or None
        return {"brand": brand, "description": description}
    except Exception:
        return {}


def synthesize_description(row: pd.Series) -> str:
    parts = [f"{row['gender_normalized']} {row['category']}"]
    if row.get("base_color"):
        parts.append(f"color {row['base_color']}")
    if row.get("usage"):
        parts.append(f"usage {row['usage']}")
    if row.get("season"):
        parts.append(f"season {row['season']}")
    return f"{row['product_name']}. " + ", ".join(parts) + "."


def build_searchable_text(row: pd.Series) -> str:
    parts = [
        row["product_name"],
        row["category"],
        row.get("base_color") or "",
        row.get("usage") or "",
        row.get("season") or "",
        row["gender"],
        row["description"],
    ]
    text = " ".join(str(p) for p in parts if p)
    return normalize_whitespace(text.lower())


# ── Main pipeline ─────────────────────────────────────────────────────────────

def build_catalog() -> pd.DataFrame:
    images_dir = DATASET_ROOT / "images"
    styles_dir = DATASET_ROOT / "styles"
    styles_csv = DATASET_ROOT / "styles.csv"

    # Step 1: load CSV
    print(f"Loading {styles_csv} ...")
    df = pd.read_csv(styles_csv, on_bad_lines="skip", dtype={"id": int})
    print(f"  {len(df):,} rows loaded")

    # Step 2: category normalization — drop unmapped rows
    df["category"] = df["articleType"].map(CATEGORY_MAP)
    df = df[df["category"].notna()].copy()
    print(f"  {len(df):,} rows in target categories")

    # Step 3: verify image files exist
    print("Verifying image files ...")
    mask = df["id"].apply(lambda i: (images_dir / f"{i}.jpg").exists())
    df = df[mask].copy()
    print(f"  {len(df):,} rows with valid images")

    # Step 4: sample per category with fixed seed
    sampled = []
    for cat, target in SAMPLE_TARGETS.items():
        subset = df[df["category"] == cat]
        if len(subset) > target:
            subset = subset.sample(n=target, random_state=RANDOM_SEED)
        sampled.append(subset)
    df = pd.concat(sampled, ignore_index=True)
    print(f"  {len(df):,} rows after sampling")
    for cat, cnt in df["category"].value_counts().sort_index().items():
        print(f"    {cat}: {cnt}")

    # Step 5: clean fields
    df["product_name"] = df["productDisplayName"].str.strip()
    df = df[df["product_name"].notna() & (df["product_name"] != "")].copy()
    df["raw_category"] = df["articleType"]
    df["subcategory"] = df["subCategory"].str.strip()
    df["gender"] = df["gender"].str.strip().fillna("Unisex")
    df["gender_normalized"] = df["gender"].replace({"Boys": "Men", "Girls": "Women"})
    df["base_color"] = df["baseColour"].str.strip().str.lower().replace("", None)
    df["usage"] = df["usage"].str.strip().replace("", None)
    df["season"] = df["season"].str.strip().replace("", None)
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")

    # Steps 6 & 7: JSON metadata — load once per product
    print(f"Loading JSON metadata for {len(df):,} products ...")
    meta: dict[int, dict] = {pid: load_json_meta(styles_dir, pid) for pid in df["id"]}

    df["brand"] = df["id"].map(lambda i: meta.get(i, {}).get("brand"))
    df["_desc_json"] = df["id"].map(lambda i: meta.get(i, {}).get("description"))
    df["description"] = df.apply(
        lambda row: row["_desc_json"] if row["_desc_json"] else synthesize_description(row),
        axis=1,
    )
    df.drop(columns=["_desc_json"], inplace=True)

    # Step 8: mock prices (deterministic, hash-based)
    df["price"] = df.apply(lambda row: mock_price(row["id"], row["category"]), axis=1)

    # Step 9: searchable_text
    df["searchable_text"] = df.apply(build_searchable_text, axis=1)

    # Step 10: image paths (relative to DATASET_ROOT)
    df["image_path"] = df["id"].apply(lambda i: f"images/{i}.jpg")

    # Step 11: final column selection
    final_cols = [
        "id", "product_name", "brand", "category", "raw_category", "subcategory",
        "gender", "gender_normalized", "base_color", "usage", "season", "year",
        "description", "image_path", "price", "searchable_text",
    ]
    df = df[final_cols].reset_index(drop=True)

    # Step 12: save
    CATALOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(CATALOG_PATH, index=False)
    print(f"\nSaved {len(df):,} products -> {CATALOG_PATH}")
    return df


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    build_catalog()
    print("\nNext step: python -m app.data.build_image_embeddings")
