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

    # Model tiering
    cheap_model: str = "gemini-1.5-flash"
    power_model: str = "gemini-1.5-pro"

    # Cost optimisation
    summary_threshold: int = 10
    use_flash_for_grounded_answers: bool = True
    response_cache_size: int = 256

    # Cognitive Memory System
    memory_top_k: int = 3
    memory_max_episodes_per_user: int = 100

    # Live Token Telemetry — USD per 1M tokens
    flash_input_usd_per_million: float = 0.075
    flash_output_usd_per_million: float = 0.30
    pro_input_usd_per_million: float = 1.25
    pro_output_usd_per_million: float = 5.00

    # Retrieval grading / rewrite / web fallback
    max_query_rewrites: int = 1
    tavily_api_key: str | None = None
    web_search_max_results: int = 5

    # Human-in-the-loop: pause on high-probability setups / trade simulations
    high_probability_threshold: float = 0.72

    class Config:
        env_file = ".env"


settings = Settings()
