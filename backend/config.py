from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # API Settings
    app_name: str = "Financial Bot API"
    debug: bool = False

    # Model Settings
    embedding_model: str = "all-MiniLM-L6-v2"
    max_results: int = 3

    # File Settings
    data_file: str = "sample_qa.json"

    # Gemini Settings (Google AI Studio: https://aistudio.google.com/app/apikey)
    gemini_api_key: str | None = None
    gemini_model: str = "gemini-1.5-flash"
    gemini_temperature: float = 0.2

    class Config:
        env_file = ".env"


settings = Settings()
