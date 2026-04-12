from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Optional
import os

class Settings(BaseSettings):
    # Alias allows pydantic to pick up DATABASE_URL from .env automatically
    database_url_env: Optional[str] = Field(None, alias="DATABASE_URL")
    
    # Critical variables should not have "working" defaults in production
    postgres_user: Optional[str] = None
    postgres_password: Optional[str] = None
    postgres_db: Optional[str] = None
    postgres_host: Optional[str] = None
    postgres_port: int = 5432
    
    openai_api_key: str = Field(..., alias="OPENAI_API_KEY")
    log_level: str = Field("INFO", alias="LOG_LEVEL")
    
    @property
    def database_url(self) -> str:
        if self.database_url_env:
            return self.database_url_env
        if all([self.postgres_user, self.postgres_password, self.postgres_host, self.postgres_db]):
            return f"postgresql://{self.postgres_user}:{self.postgres_password}@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        return "" # Fallback to empty if not configured

    class Config:
        env_file = ".env"
        env_file_encoding = 'utf-8'
        extra = "ignore"
        populate_by_name = True

settings = Settings()
