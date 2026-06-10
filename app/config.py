from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    anthropic_api_key: str
    openai_api_key: str
    database_url: str
    embedding_model: str = "text-embedding-3-small"
    chat_model: str = "claude-haiku-4-5-20251001"
    chunk_size: int = 1000
    chunk_overlap: int = 150
    retrieval_k: int = 4

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()
