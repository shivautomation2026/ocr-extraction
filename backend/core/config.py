from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict
import os
from typing import Optional

class Settings(BaseSettings):
    MONGODB_URI: str = ""
    MISTRAL_API_KEY: str = ""
    GOOGLE_API_KEY: str = ""
    HF_API_KEY: str = ""
    BASE_URL: str = ""
    COMPANY_DB: str = ""
    USERNAME: str = ""
    PASSWORD: str = ""

    GEMINI_API_KEY: str = ""
    GEMINI_MODEL_NAME: str = ""

    PINECONE_API_KEY: Optional[str] = None
    PINECONE_INDEX_NAME: str = ""

    EMBEDDING_MODEL: str = "sentence-transformers/all-mpnet-base-v2"
    TOP_K: int = 10
    ALPHA: float = 0.5

    GOOGLE_CLOUD_PROJECT: str = ""
    GOOGLE_CLOUD_LOCATION: str = ""
    GOOGLE_APPLICATION_CREDENTIALS: str = ""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore"
    )

settings = Settings()