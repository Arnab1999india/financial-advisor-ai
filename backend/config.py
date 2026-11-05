from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # API Settings
    app_name: str = "Financial Bot API"
    debug: bool = False

    # Model Settings
    embedding_model: str = "all-MiniLM-L6-v2"
    max_results: int = 4  # legacy alias; prefer retrieval_final_k

    # RAG retrieval / grounding
    retrieval_fetch_k: int = 5
    retrieval_final_k: int = 4
    chunk_max_chars: int = 1200
    retrieval_min_confidence: float = 0.38
    hybrid_bm25_enabled: bool = True
    hybrid_bm25_weight: float = 0.35
    grounding_full_confidence: float = 0.52
    grounding_policy_file: str = "grounding_policy.json"
    strict_two_step_kb_only: bool = True
    three_step_ambiguity_check: bool = True

    # File Settings
    data_file: str = "sample_qa.json"
    domain_guardrails_file: str = "domain_guardrails.json"
    security_policy_file: str = "security_policy.json"

    # Input validation
    query_min_length: int = 2
    query_max_length: int = 1000
    upload_max_bytes: int = 5_000_000
    upload_max_pdf_pages: int = 50
    max_conversation_turns: int = 40
    abuse_score_threshold: int = 5
    abuse_cooldown_seconds: int = 900

    # Rate limiting (in-memory fallback; optional Redis via rate_limit_redis_url)
    rate_limit_enabled: bool = True
    rate_limit_redis_url: str | None = None
    rate_limit_rpm_per_ip: int = 20
    rate_limit_rpm_per_user: int = 20
    rate_limit_rph_per_ip: int = 200
    rate_limit_rph_per_user: int = 200
    rate_limit_daily_tokens_per_user: int = 500_000
    rate_limit_upload_rpm: int = 5
    rate_limit_web_rpm: int = 10
    rate_limit_max_inflight_per_user: int = 2
    rate_limit_global_inflight: int = 20

    # Resilience / timeouts (seconds)
    external_default_timeout_seconds: float = 30.0
    llm_timeout_seconds: float = 45.0
    embedding_timeout_seconds: float = 20.0
    market_data_timeout_seconds: float = 15.0
    web_search_timeout_seconds: float = 12.0
    external_retry_max_attempts: int = 3
    external_retry_base_seconds: float = 0.4
    circuit_breaker_failure_threshold: int = 5
    circuit_breaker_open_seconds: int = 60

    # Caching TTLs (seconds)
    exact_cache_ttl_seconds: int = 900
    semantic_cache_ttl_seconds: int = 900
    semantic_cache_similarity_threshold: float = 0.92
    semantic_cache_max_entries: int = 256
    document_answer_cache_ttl_seconds: int = 3600
    market_data_cache_ttl_seconds: int = 120

    # LLM generation caps
    llm_max_output_tokens_flash: int = 768
    llm_max_output_tokens_pro: int = 1024
    llm_stop_sequences: list[str] = Field(default_factory=lambda: ["<context_data>", "<tool_data>"])

    # Cost alerts (USD, daily)
    cost_alert_global_usd_daily: float = 25.0
    cost_alert_user_usd_daily: float = 2.0

    # Gemini Settings (Google AI Studio: https://aistudio.google.com/app/apikey)
    gemini_api_key: str | None = None
    gemini_model: str = "gemini-2.0-flash"
    gemini_temperature: float = 0.1

    # Model tiering
    cheap_model: str = "gemini-2.0-flash"
    power_model: str = "gemini-2.0-flash"

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
    market_data_freshness_minutes: int = 15

    # Human-in-the-loop: pause on high-probability setups / trade simulations
    high_probability_threshold: float = 0.72

    class Config:
        env_file = ".env"


settings = Settings()
