"""
Offline script: generate CLIP embeddings for all catalog images.

Usage:
  python -m app.data.build_image_embeddings

Reads:  data/processed/catalog.parquet
Reads:  {DATASET_ROOT}/{image_path} per row (e.g. images/24050.jpg)
Writes: data/processed/image_embeddings.npy   shape [N, 512] float32 L2-normalized
Writes: data/processed/embedding_ids.npy      shape [N] int64

Safe to re-run: overwrites existing files.
"""

import sys

import numpy as np
import pandas as pd

from app.config import CATALOG_PATH, DATASET_ROOT, EMBEDDING_IDS_PATH, EMBEDDINGS_PATH


def main() -> None:
    if not CATALOG_PATH.exists():
        print(f"ERROR: catalog not found at {CATALOG_PATH}")
        print("Run: python -m app.data.preprocess_dataset")
        sys.exit(1)

    try:
        import torch
        from PIL import Image
        from transformers import CLIPModel, CLIPProcessor
    except ImportError as e:
        print(f"ERROR: missing dependency: {e}")
        print("Run: pip install torch transformers Pillow")
        sys.exit(1)

    df = pd.read_parquet(CATALOG_PATH)
    print(f"Loaded catalog: {len(df):,} products")

    model_name = "openai/clip-vit-base-patch32"
    print(f"Loading CLIP model ({model_name}) ...")
    processor = CLIPProcessor.from_pretrained(model_name)
    model = CLIPModel.from_pretrained(model_name)
    model.eval()

    all_vecs: list[np.ndarray] = []
    all_ids: list[int] = []
    skipped = 0
    batch_size = 16
    rows = df.to_dict("records")

    print(f"Encoding {len(rows):,} images (CPU, batch_size={batch_size}) ...")
    for i in range(0, len(rows), batch_size):
        batch = rows[i : i + batch_size]
        images, valid_ids = [], []

        for row in batch:
            img_path = DATASET_ROOT / row["image_path"]
            try:
                images.append(Image.open(img_path).convert("RGB"))
                valid_ids.append(row["id"])
            except Exception:
                skipped += 1

        if not images:
            continue

        inputs = processor(images=images, return_tensors="pt", padding=True)
        with torch.no_grad():
            vecs = model.get_image_features(**inputs)
            vecs = vecs / vecs.norm(dim=-1, keepdim=True)  # L2-normalize

        all_vecs.append(vecs.numpy().astype(np.float32))
        all_ids.extend(valid_ids)

        done = min(i + batch_size, len(rows))
        if done % 200 == 0 or done == len(rows):
            print(f"  {done}/{len(rows)}")

    embeddings_np = np.vstack(all_vecs)
    ids_np = np.array(all_ids, dtype=np.int64)

    EMBEDDINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.save(EMBEDDINGS_PATH, embeddings_np)
    np.save(EMBEDDING_IDS_PATH, ids_np)

    print(f"Done. Embedded {len(all_ids):,} images, skipped {skipped}.")
    print(f"  embeddings -> {EMBEDDINGS_PATH}  shape={embeddings_np.shape}")
    print(f"  ids        -> {EMBEDDING_IDS_PATH}  shape={ids_np.shape}")


if __name__ == "__main__":
    main()
