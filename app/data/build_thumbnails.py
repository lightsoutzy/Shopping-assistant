"""
Generates resized thumbnails for all catalog products.

Usage:
  python -m app.data.build_thumbnails

Reads:  data/processed/catalog.parquet
Reads:  {DATASET_ROOT}/{image_path} per row
Writes: data/catalog_thumbnails/{id}.jpg  (300x400, JPEG quality 75)

Output directory is committed to the repo so Railway and Streamlit Cloud
can serve product images without access to the raw dataset.
Safe to re-run: overwrites existing files.
"""

import sys
from pathlib import Path

import pandas as pd
from PIL import Image

from app.config import CATALOG_PATH, DATASET_ROOT

THUMB_DIR = Path("data/catalog_thumbnails")
TARGET_W, TARGET_H = 300, 400
JPEG_QUALITY = 75


def make_thumbnail(src: Path, dst: Path) -> bool:
    """Resize to fit within TARGET_W x TARGET_H, pad with white, save as JPEG."""
    try:
        img = Image.open(src).convert("RGB")
        img.thumbnail((TARGET_W, TARGET_H), Image.LANCZOS)
        # Centre on a white canvas so all thumbnails are the same size
        canvas = Image.new("RGB", (TARGET_W, TARGET_H), (255, 255, 255))
        x = (TARGET_W - img.width) // 2
        y = (TARGET_H - img.height) // 2
        canvas.paste(img, (x, y))
        canvas.save(dst, "JPEG", quality=JPEG_QUALITY, optimize=True)
        return True
    except Exception as e:
        print(f"  Skipped {src.name}: {e}")
        return False


def main() -> None:
    if not CATALOG_PATH.exists():
        print(f"ERROR: catalog not found at {CATALOG_PATH}")
        print("Run: python -m app.data.preprocess_dataset")
        sys.exit(1)

    THUMB_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(CATALOG_PATH)
    print(f"Generating thumbnails for {len(df):,} products -> {THUMB_DIR}")

    ok = skipped = 0
    for _, row in df.iterrows():
        src = DATASET_ROOT / row["image_path"]
        dst = THUMB_DIR / f"{row['id']}.jpg"
        if make_thumbnail(src, dst):
            ok += 1
        else:
            skipped += 1

    total_mb = sum(p.stat().st_size for p in THUMB_DIR.glob("*.jpg")) / 1024 / 1024
    print(f"Done. {ok:,} thumbnails, {skipped} skipped. Total size: {total_mb:.1f} MB")


if __name__ == "__main__":
    main()
