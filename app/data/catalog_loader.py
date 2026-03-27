"""
Loads the preprocessed catalog and embedding files into memory at startup.
Called once from app/main.py; results are shared across all requests.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from app.config import CATALOG_PATH, EMBEDDING_IDS_PATH, EMBEDDINGS_PATH


@dataclass
class CatalogStore:
    df: pd.DataFrame               # normalized product catalog
    embeddings: np.ndarray | None  # shape [N, 512], float32, L2-normalized; None if not built
    embedding_ids: np.ndarray | None  # shape [N], int64, parallel to embeddings


def load_catalog() -> CatalogStore:
    """Load catalog.parquet and optional embedding .npy files from disk."""
    if not CATALOG_PATH.exists():
        raise FileNotFoundError(
            f"Catalog not found at {CATALOG_PATH}. "
            "Run: python -m app.data.preprocess_dataset"
        )

    df = pd.read_parquet(CATALOG_PATH)

    embeddings: np.ndarray | None = None
    embedding_ids: np.ndarray | None = None

    if EMBEDDINGS_PATH.exists() and EMBEDDING_IDS_PATH.exists():
        embeddings = np.load(EMBEDDINGS_PATH)
        embedding_ids = np.load(EMBEDDING_IDS_PATH)
    else:
        print(
            "Warning: embedding files not found. Image search will be unavailable. "
            "Run: python -m app.data.preprocess_dataset --embeddings"
        )

    return CatalogStore(df=df, embeddings=embeddings, embedding_ids=embedding_ids)
