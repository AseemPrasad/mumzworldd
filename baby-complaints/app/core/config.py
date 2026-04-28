# app/core/config.py
# Developer note: Central configuration loaded from .env via python-dotenv.
# Extend by adding new fields here and referencing them via `settings` singleton.

from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    DATA_PATH: str = "./app/data/products_sample.csv"
    OPENROUTER_API_KEY: str = ""
    OPENROUTER_API_URL: str = "https://openrouter.ai/api/v1/chat/completions"
    OPENROUTER_MODEL: str = "openai/gpt-3.5-turbo"
    APP_VERSION: str = "0.1.0"
    LOG_LEVEL: str = "INFO"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings() -> Settings:
    return Settings()
