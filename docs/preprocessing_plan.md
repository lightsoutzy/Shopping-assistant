# Preprocessing Plan

## Goal

Transform the raw Kaggle Fashion Product Images Dataset into:
1. `data/processed/catalog.parquet` — normalized product catalog (~1,800 rows)
2. `data/processed/image_embeddings.npy` — CLIP image embeddings for all catalog products
3. `data/processed/embedding_ids.npy` — product IDs parallel to the embeddings array

These files are generated once offline and loaded at runtime. They are gitignored.

---

## Script Location

```
app/data/preprocess_dataset.py   ← main preprocessing script (implemented in Phase 2)
```

Run manually before starting the server:
```bash
python -m app.data.preprocess_dataset
```

---

## Step-by-Step Flow

### Step 1 — Load styles.csv

```
Input:  {DATASET_ROOT}/styles.csv
Action: pd.read_csv(..., on_bad_lines='skip')
Notes:  The file has ~44,446 rows and some malformed lines; skip bad rows.
        Columns used: id, gender, subCategory, articleType, baseColour,
                      season, year, usage, productDisplayName
```

### Step 2 — Apply Category Normalization Map

```
Action: Map articleType → demo category using the normalization map in schema.md.
        Drop rows where articleType is not in the map.
Result: ~6,000+ candidate rows across the 8 demo categories.

Category map (articleType → category):
  Tshirts       → t-shirt
  Shirts        → shirt
  Casual Shoes  → shoes
  Formal Shoes  → shoes
  Flats         → shoes
  Sports Shoes  → sneakers
  Jackets       → jacket
  Sweatshirts   → hoodie
  Sweaters      → hoodie
  Shorts        → shorts
  Handbags      → bag
  Backpacks     → bag
  Clutches      → bag
```

### Step 3 — Verify Image Files Exist

```
Action: For each row, check {DATASET_ROOT}/images/{id}.jpg exists on disk.
        Drop rows where the image is missing.
Notes:  The dataset has a known mismatch between styles.csv entries and
        available images. Typically ~20% of rows have no image.
        This step is slow (~seconds); progress bar recommended.
```

### Step 4 — Sample to Target Counts

```
Action: For each category, sample rows using a fixed random seed (seed=42)
        so the catalog is reproducible across runs.

Target counts per category:
  t-shirt   → 250
  shirt     → 200
  shoes     → 200
  sneakers  → 200
  jacket    → all available (expected ~200–258)
  hoodie    → all available (expected ~250–300)
  shorts    → 200
  bag       → 200

If a category has fewer rows than the target, keep all of them.
Final catalog size: ~1,700–1,800 rows.
```

### Step 5 — Clean and Normalize Fields

```
product_name:  strip(), drop rows where empty after strip
gender:        strip(); map Boys→Men, Girls→Women for gender_normalized column;
               default Unisex if missing
base_color:    strip().lower(); set None if empty string
usage:         strip(); set None if empty
season:        strip(); set None if empty
year:          coerce to int; set None on failure
```

### Step 6 — Load Brand from JSON Metadata (best-effort)

```
Action: For each product id, attempt to read {DATASET_ROOT}/styles/{id}.json
        Extract data.brandName if present.
        If file missing or key absent, brand = None.
Notes:  This is optional enrichment. Do not fail preprocessing if JSON is missing.
        Loading ~1,800 JSON files takes a few seconds; acceptable.
```

### Step 7 — Load Description from JSON Metadata (best-effort)

```
Action: For each product id, attempt to read {DATASET_ROOT}/styles/{id}.json
        (reuse the loaded JSON from Step 6 — load each file only once)
        Extract data.productDescriptors.description.value
        Strip HTML tags using a simple regex or html.parser.
        If unavailable, fall back to synthesized description:
          "{product_name}. {gender_normalized} {category}. Color: {base_color}. Usage: {usage}. Season: {season}."
```

### Step 8 — Generate Mock Prices

```python
CATEGORY_MULTIPLIERS = {
    "bag": 3.0, "jacket": 4.0, "shoes": 3.5, "sneakers": 3.0,
    "hoodie": 2.5, "shirt": 2.0, "t-shirt": 1.5, "shorts": 1.5,
}
base = (id % 10) * 5 + 20
price = round(base * CATEGORY_MULTIPLIERS[category], 2)
```

Same id always produces the same price across runs.

### Step 9 — Build searchable_text

```python
searchable_text = f"{product_name} {category} {base_color or ''} {usage or ''} {season or ''} {gender} {description}"
searchable_text = " ".join(searchable_text.lower().split())  # normalize whitespace
```

### Step 10 — Build image_path Column

```python
image_path = f"images/{id}.jpg"  # relative to DATASET_ROOT
```

### Step 11 — Select and Order Final Columns

```
Final DataFrame columns (in order):
  id, product_name, brand, category, raw_category, subcategory,
  gender, gender_normalized, base_color, usage, season, year,
  description, image_path, price, searchable_text
```

### Step 12 — Save Catalog

```
Output: data/processed/catalog.parquet
Action: df.to_parquet(..., index=False)
Log:    print row counts per category and total
```

---

## Offline Image Embedding Step (Phase 3, separate script)

This step is separate from catalog preprocessing because it takes longer and requires CLIP.

```
Script: app/data/preprocess_dataset.py --embeddings  (or a separate flag)

For each row in catalog.parquet:
  1. Load image from {DATASET_ROOT}/{image_path}
  2. Preprocess with CLIP processor
  3. Encode with CLIP vision encoder (CPU)
  4. Store 512-dim float32 vector

Output:
  data/processed/image_embeddings.npy  — shape [N, 512]
  data/processed/embedding_ids.npy     — shape [N], int64

Notes:
  - ~1,800 images × ~0.1s each on CPU ≈ 3–5 minutes total
  - Use batch processing (batch size 16) to reduce overhead
  - Skip images that fail to load; log skipped IDs
  - Embeddings are L2-normalized before saving (unit vectors → dot product = cosine similarity)
```

---

## Expected Outputs

| File                              | Size (approx) | Contents                            |
|-----------------------------------|---------------|-------------------------------------|
| `data/processed/catalog.parquet`  | ~2–5 MB       | ~1,800 product rows, all fields     |
| `data/processed/image_embeddings.npy` | ~3.5 MB   | 1,800 × 512 float32 vectors         |
| `data/processed/embedding_ids.npy`   | ~15 KB    | 1,800 int64 IDs                     |

---

## Known Dataset Issues to Handle

| Issue | How handled |
|-------|-------------|
| ~20% of styles.csv rows have no matching image | Dropped in Step 3 |
| Malformed rows in styles.csv | `on_bad_lines='skip'` in pd.read_csv |
| JSON metadata missing for some products | Silent fallback to synthesized description |
| Category names in dataset don't match demo names | Normalization map in Step 2 |
| `year` column has non-numeric values | Coerce to int, None on failure |
| Some `baseColour` values are multi-word (e.g. "Navy Blue") | Kept as-is; lowercased |
| `gender` values include "Boys" and "Girls" | Normalized in Step 5 |
