import os
from pathlib import Path

from dotenv import load_dotenv

# Resolve the project root absolutely from this file's location.
# config.py lives at <project_root>/app/config.py → parent.parent = project root.
# This works regardless of the CWD when uvicorn/python is started.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent

_env = _PROJECT_ROOT / ".env"
_env_txt = _PROJECT_ROOT / ".env.txt"

if _env.exists():
    load_dotenv(_env)
    print(f"[CONFIG] Loaded {_env}", flush=True)
elif _env_txt.exists():
    load_dotenv(_env_txt)
    print(f"[CONFIG] Loaded {_env_txt} — consider renaming to .env", flush=True)
else:
    load_dotenv()  # last resort: python-dotenv standard search
    print(f"[CONFIG] No .env or .env.txt found in {_PROJECT_ROOT} — relying on OS env", flush=True)

DATASET_ROOT = Path(os.getenv("DATASET_ROOT", "data/fashion-dataset/fashion-dataset"))
CATALOG_PATH = Path(os.getenv("CATALOG_PATH", "data/processed/catalog.parquet"))
EMBEDDINGS_PATH = Path(os.getenv("EMBEDDINGS_PATH", "data/processed/image_embeddings.npy"))
EMBEDDING_IDS_PATH = Path(os.getenv("EMBEDDING_IDS_PATH", "data/processed/embedding_ids.npy"))

ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
MAX_REQUESTS_PER_SESSION: int = int(os.getenv("MAX_REQUESTS_PER_SESSION", "100"))
MAX_TOTAL_REQUESTS: int = int(os.getenv("MAX_TOTAL_REQUESTS", "2000"))
API_HOST: str = os.getenv("API_HOST", "localhost")
API_PORT: int = int(os.getenv("API_PORT", "8000"))
ORCHESTRATOR_TEMPERATURE: float = float(os.getenv("ORCHESTRATOR_TEMPERATURE", "0.6"))
TRANSCRIPT_WINDOW: int = int(os.getenv("TRANSCRIPT_WINDOW", "10"))
