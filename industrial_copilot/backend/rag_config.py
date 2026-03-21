from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    # Replaced localized credentials with a universal Neon DB capable URL format
    database_url: str = "postgresql://your_neon_user:your_neon_password@ep-your-neon-endpoint.neon.tech/neondb?sslmode=require"
    
    openai_api_key: str = ""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

settings = Settings()
