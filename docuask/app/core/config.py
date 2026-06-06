from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    groq_api_key: str
    database_url: str
    secret_key: str = "dev-secret-change-in-production"
    environment: str = "development"

    chunk_size: int = 1000
    chunk_overlap: int = 200
    top_k_chunks: int = 5

    # Groq models
    embedding_model: str = "text-embedding-3-small"   # we use OpenAI embeddings (free tier)
    chat_model: str = "llama-3.1-8b-instant"                  # Groq's free fast model

    class Config:
        env_file = ".env"


@lru_cache()
def get_settings() -> Settings:
    return Settings()