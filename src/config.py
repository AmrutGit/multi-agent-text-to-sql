from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    db_path: str = "data/northwind.sqlite"
    llm_model: str = "gemini-1.5-flash"
    google_api_key: str = ""
    max_sql_retries: int = 2  # validator -> sql_writer self-correction loop


settings = Settings()
