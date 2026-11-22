from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    db_path: str = "data/northwind.sqlite"
    max_sql_retries: int = 2  # validator -> sql_writer self-correction loop
    max_rows: int = 200  # hard cap on rows returned by /query
    query_timeout_s: float = 5.0  # abort runaway queries (e.g. accidental cross joins)
    explain_row_limit: int = 20  # rows shown to the explainer LLM
    schema_mode: str = "auto"  # "auto" | "retrieve" (RAG slice) | "full" (whole schema)
    schema_budget_chars: int = 12000  # auto: retrieve only when full schema exceeds this (~3k tokens)

    # Default LLM (the API serves /query; benchmarks pick models from evals/models.yaml)
    llm_provider: str = "gemini"  # gemini | anthropic | openai_compat
    llm_model: str = "gemini-3.8-flash"
    llm_base_url: str = ""  # openai_compat only, e.g. http://localhost:8001/v1
    llm_api_key_env: str = "GEMINI_API_KEY"  # NAME of the env var holding the key (GOOGLE_API_KEY also accepted)
    google_api_key: str = ""  # kept for .env convenience / backwards compatibility


settings = Settings()
