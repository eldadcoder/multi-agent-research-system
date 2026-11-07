"""Typed settings loaded from environment variables / .env."""
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # Ollama
    ollama_base_url: str = "http://localhost:11434"
    reasoning_model: str = "qwen2.5:7b"
    fast_model: str = "qwen2.5:3b"
    embed_model: str = "nomic-embed-text"
    num_ctx: int = 8192
    temperature: float = 0.2

    # LLM call behaviour
    llm_timeout_s: float = 300.0
    llm_max_attempts: int = 3
    max_parallel_llm_calls: int = 1  # one local GPU/CPU: keep this low

    # Pipeline limits
    max_iterations: int = 2

    # Search
    tavily_api_key: str = ""


@lru_cache
def get_settings() -> Settings:
    return Settings()