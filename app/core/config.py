import os
from functools import lru_cache
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from qdrant_client import QdrantClient  # ──► 1. Added import

# Dynamically resolve the absolute system workspace root path directory
BASE_DIR = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    app_name: str = "Enterprise IT Support Agentic RAG Copilot"
    app_env: str = "development"

    # 🌲 LOCAL OLLAMA HARDWARE CONFIGURATIONS
    llm_model: str = "qwen2.5:7b"
    # llm_model: str = "qwen2.5:14b"
    embedding_model: str = "bge-m3"

    # 📦 LOCAL STORAGE VAULT SETTINGS
    qdrant_path: str = str(BASE_DIR / "qdrant_storage")
    collection_name: str = "enterprise_it_kb"  # Set fallback default string
    lectures_folder: str = "/Users/surendersinghthakran/Masai-AI ML/Lectures"

    # 🌍 CLOUD CONNECTIONS AND SECURE ACCESS TOKENS
    tavily_api_key: str = ""
    admin_api_key: str = ""

    # 🗃️ COMPONENT ENGINE PATH PARAMETERS
    top_k: int = 100
    max_retries: int = 1
    audit_db_path: str = str(BASE_DIR / "data" / "audit.db")
    upload_dir: str = str(BASE_DIR / "uploads")
    sample_kb_dir: str = str(BASE_DIR / "data" / "sample_kb")

    # Enforce environment configuration mappings
    model_config = SettingsConfigDict(env_file=str(BASE_DIR / ".env"), extra="ignore")


@lru_cache
def get_settings() -> Settings:
    """Returns a memoized singleton instance of app settings configurations."""
    return Settings()


# Instantiate the global settings dictionary for backward-compatible access loops
settings = get_settings()

# ──► 2. INSTANTIATE THE SINGLE CENTRAL GLOBAL CLIENT POOL INSTANCE HERE
# This opens the storage file once and shares the pipeline across ingest and retrieval
os.makedirs(settings.qdrant_path, exist_ok=True)
qdrant_client = QdrantClient(path=settings.qdrant_path)
