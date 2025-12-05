from pydantic_settings import BaseSettings
from typing import Literal, Optional


class Settings(BaseSettings):
    LLM_PROVIDER: Literal["huggingface", "ollama"] = "huggingface"
    HF_TOKEN: Optional[str] = None
    HF_MODEL: str = "google/gemma-2b-it:fastest"
    OLLAMA_MODEL: str = "llama3.2:1b"  # Only used if LLM_PROVIDER="ollama"

    DATA_DIR: str = "data"
    UPLOAD_DIR: str = "data/uploads"
    INDEX_PATH: str = "data/faiss.index"
    META_PATH: str = "data/metadata.jsonl"

    model_config = {"env_file": ".env", "extra": "ignore"}


settings = Settings()
# ====================================================================
# Purpose

# Centralized config using pydantic-settings
# Loads from .env file
# No hardcoding of paths or tokens
# Type-safe with Literal and Optional