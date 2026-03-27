# Normalized Product Schema

## Runtime Catalog Format

The catalog is stored as `data/processed/catalog.parquet`.
At runtime it is loaded once into a pandas DataFrame and kept in memory.

---

## Field Definitions

### `id` — int
- **Source:** `styles.csv` column `id`
- **Notes:** Primary key. Also the image filename stem (`{id}.jpg`) and the input to deterministic price hashing.

### `product_name` — str
- **Source:** `styles.csv` column `productDisplayName`
- **Notes:** Stripped of leading/trailing whitespace. Never null; rows with empty names are dropped during preprocessing.

### `brand` — str | None
- **Source:** `styles/{id}.json` → `data.brandName`
- **Notes:** Available in JSON metadata for most products. Set to `null` if JSON is missing or `brandName` is absent. Not used in retrieval; shown in response only.

### `category` — str
- **Source:** Normalized from `styles.csv` column `articleType` using the category map (see below).
- **Values:** one of `t-shirt`, `shirt`, `shoes`, `sneakers`, `jacket`, `hoodie`, `shorts`, `bag`
- **Notes:** This is the demo-facing category. Used in filtering and ranking.

### `raw_category` — str
- **Source:** `styles.csv` column `articleType`, original value.
- **Notes:** Kept for debugging and potential future category expansion. Not used in retrieval.

### `subcategory` — str | None
- **Source:** `styles.csv` column `subCategory`
- **Values:** e.g., `Topwear`, `Footwear`, `Bags`
- **Notes:** Coarse grouping from the dataset. Nullable.

### `gender` — str
- **Source:** `styles.csv` column `gender`
- **Raw values:** `Men`, `Women`, `Unisex`, `Boys`, `Girls`
- **Normalization:** `Boys` → `Men`; `Girls` → `Women` for filtering logic. Raw value is stored in `gender`; `gender_normalized` is a derived column used in ranker.
- **Notes:** Never null in this dataset; default to `Unisex` if missing.

### `base_color` — str | None
- **Source:** `styles.csv` column `baseColour`
- **Notes:** Lowercased and stripped. Common values: `black`, `white`, `blue`, `red`, `grey`, etc. Null if missing.

### `usage` — str | None
- **Source:** `styles.csv` column `usage`
- **Values:** `Casual`, `Sports`, `Formal`, `Ethnic`, `Smart Casual`, `Travel`, `Party`
- **Notes:** Nullable. One of the most useful filter fields for recommendation.

### `season` — str | None
- **Source:** `styles.csv` column `season`
- **Values:** `Summer`, `Winter`, `Fall`, `Spring`
- **Notes:** Nullable. Used in soft ranking, not hard filtering.

### `year` — int | None
- **Source:** `styles.csv` column `year`
- **Notes:** Nullable. Not used in retrieval or ranking for this demo; kept for completeness.

### `description` — str
- **Source (preferred):** `styles/{id}.json` → `data.productDescriptors.description.value`, HTML-stripped
- **Source (fallback):** Synthesized from structured fields:
  ```
  "{product_name}. {gender} {category}. Color: {base_color}. Usage: {usage}. Season: {season}."
  ```
- **Notes:** Only real catalog facts are used. Never invented. HTML tags stripped if sourced from JSON.

### `image_path` — str
- **Source:** `images/{id}.jpg` relative to `DATASET_ROOT`
- **Notes:** Verified to exist during preprocessing. Rows where the image file is missing are dropped.

### `price` — float
- **Source:** Deterministic mock — not from the dataset.
  - The JSON does contain an INR price (`data.price`), but it is not meaningful for a USD-denominated demo.
- **Formula:**
  ```python
  CATEGORY_MULTIPLIERS = {
      "bag":     3.0,
      "jacket":  4.0,
      "shoes":   3.5,
      "sneakers":3.0,
      "hoodie":  2.5,
      "shirt":   2.0,
      "t-shirt": 1.5,
      "shorts":  1.5,
  }
  base = (id % 10) * 5 + 20          # base: $20–$65 in $5 steps
  price = round(base * multiplier, 2)
  ```
- **Range:** ~$30 (t-shirt, id % 10 == 0) to ~$260 (jacket, id % 10 == 9). Typical range $30–$200.
- **Notes:** Same `id` always produces the same price. Price filtering in the ranker uses this value.

### `searchable_text` — str
- **Source:** Constructed at preprocessing time from structured fields.
- **Formula:**
  ```python
  f"{product_name} {category} {base_color or ''} {usage or ''} {season or ''} {gender} {description}"
  ```
  Lowercased and whitespace-normalized.
- **Notes:** This is the field fed to the TF-IDF vectorizer. It is not stored redundantly — it is reconstructed during the preprocess step and stored in the parquet file for fast loading.

---

## Category Normalization Map

The dataset `articleType` values do not match the demo category names.
This map is applied during preprocessing:

| Dataset `articleType`   | Demo `category` |
|-------------------------|-----------------|
| `Tshirts`               | `t-shirt`       |
| `Shirts`                | `shirt`         |
| `Casual Shoes`          | `shoes`         |
| `Formal Shoes`          | `shoes`         |
| `Flats`                 | `shoes`         |
| `Sports Shoes`          | `sneakers`      |
| `Jackets`               | `jacket`        |
| `Sweatshirts`           | `hoodie`        |
| `Sweaters`              | `hoodie`        |
| `Shorts`                | `shorts`        |
| `Handbags`              | `bag`           |
| `Backpacks`             | `bag`           |
| `Clutches`              | `bag`           |

Any `articleType` not in this map is excluded from the catalog.

---

## Approximate Catalog Composition (target ~1,800 products)

| Demo category | Dataset source              | Target count |
|---------------|-----------------------------|--------------|
| t-shirt       | Tshirts (7,070 available)   | 250          |
| shirt         | Shirts (3,217 available)    | 200          |
| shoes         | Casual + Formal + Flats     | 200          |
| sneakers      | Sports Shoes (2,036)        | 200          |
| jacket        | Jackets (258 available)     | 258 (all)    |
| hoodie        | Sweatshirts + Sweaters      | 285+         |
| shorts        | Shorts (547 available)      | 200          |
| bag           | Handbags + Backpacks + Clutches | 200      |
| **Total**     |                             | **~1,793**   |

For overrepresented categories (t-shirt, shirt), a random sample with fixed seed is taken to reach the target count.

---

## Embedding Files

These files live in `data/processed/` and are generated by the offline embedding script (Phase 3):

| File                     | Shape       | Notes                                             |
|--------------------------|-------------|---------------------------------------------------|
| `image_embeddings.npy`   | [N, 512]    | CLIP clip-vit-base-patch32 embeddings, float32    |
| `embedding_ids.npy`      | [N]         | Product IDs corresponding to each embedding row  |

N = number of catalog products with a valid image file.
The IDs array is used to map similarity scores back to catalog rows.
