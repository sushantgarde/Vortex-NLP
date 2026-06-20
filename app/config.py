"""
app/config.py

Centralized application configuration, loaded from environment
variables / .env via pydantic-settings. Import `settings` anywhere
you need a path or a knob — don't hardcode paths elsewhere.
"""
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # --- Ollama ---
    OLLAMA_HOST: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "llama3.1:8b"
    OLLAMA_TIMEOUT_SECONDS: int = 120

    # --- App / server ---
    APP_HOST: str = "0.0.0.0"
    APP_PORT: int = 8000
    FRONTEND_PORT: int = 8501
    APP_NAME: str = "The Auditor"
    APP_VERSION: str = "0.1.0"

    # --- Paths ---
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    DATA_RAW_DIR: Path = BASE_DIR / "data" / "raw"
    DATA_PROCESSED_DIR: Path = BASE_DIR / "data" / "processed"
    DATA_SYNTHETIC_DIR: Path = BASE_DIR / "data" / "synthetic"
    EMBEDDING_INDEX_DIR: Path = BASE_DIR / "models" / "embedding_index"
    CLASSIFIER_MODEL_DIR: Path = BASE_DIR / "models" / "classifier"

    GUIDELINE_CHUNKS_PATH: Path = DATA_PROCESSED_DIR / "guideline_chunks.json"
    PROTOCOL_STATEMENTS_PATH: Path = DATA_PROCESSED_DIR / "protocol_statements.json"
    FAISS_INDEX_PATH: Path = EMBEDDING_INDEX_DIR / "guideline.index"
    FAISS_METADATA_PATH: Path = EMBEDDING_INDEX_DIR / "guideline_meta.json"

    # --- Embeddings / retrieval ---
    EMBEDDING_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"
    RETRIEVAL_TOP_K: int = 5

    # --- Classifier (optional path per your tree) ---
    USE_CLASSIFIER: bool = False

    # --- Database ---
    DATABASE_URL: str = "sqlite:///./data/processed/auditor.db"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    def ensure_directories(self) -> None:
        for path in [
            self.DATA_RAW_DIR,
            self.DATA_PROCESSED_DIR,
            self.DATA_SYNTHETIC_DIR,
            self.EMBEDDING_INDEX_DIR,
            self.CLASSIFIER_MODEL_DIR,
        ]:
            path.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    s = Settings()
    s.ensure_directories()
    return s


settings = get_settings()