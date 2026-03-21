from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Optional
import os

class Settings(BaseSettings):
    # Alias allows pydantic to pick up DATABASE_URL from .env automatically
    database_url_env: Optional[str] = Field(None, alias="DATABASE_URL")
    
    postgres_user: str = "myuser"
    postgres_password: str = "mypassword"
    postgres_db: str = "rag_db"
    postgres_host: str = "localhost"
    postgres_port: int = 5433
    
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    
    @property
    def database_url(self) -> str:
        if self.database_url_env:
            return self.database_url_env
        return f"postgresql://{self.postgres_user}:{self.postgres_password}@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"

    class Config:
        env_file = ".env"
        env_file_encoding = 'utf-8'
        extra = "ignore"
        populate_by_name = True

settings = Settings()
