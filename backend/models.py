from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any


class QueryRequest(BaseModel):
    query: str
    session_id: Optional[str] = None  # Reserved for multi-turn conversation support
    user_id: Optional[str] = None     # Enables Cognitive Memory System when provided


class TokenCallEvent(BaseModel):
    node: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    estimated_cost_usd: float = 0.0
    recorded_at: Optional[str] = None


class TokenUsageSummary(BaseModel):
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    estimated_cost_usd: float = 0.0
    by_node: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    events: List[TokenCallEvent] = Field(default_factory=list)


class SourceChunk(BaseModel):
    chunk_id: str
    title: str = ""
    snippet: str = ""
    category: str = "general"
    confidence: float = 0.0
    retrieved_at: Optional[str] = None
    source_timestamp: Optional[str] = None
    source_url: Optional[str] = None
    source_type: str = "faiss"


class SentenceMapping(BaseModel):
    answer_sentence: str
    answer_start: int
    answer_end: int
    chunk_id: str = ""
    source_title: str = ""
    source_sentence: str = ""
    source_type: str = "faiss"
    confidence: float = 0.0
    source_timestamp: Optional[str] = None


class ApprovalCard(BaseModel):
    reason: str
    draft_answer: str = ""
    proposal: Dict[str, Any] = Field(default_factory=dict)


class ApprovalResumeRequest(BaseModel):
    thread_id: str
    decision: str  # "approve" | "reject"
    params: Optional[Dict[str, Any]] = None
    notes: Optional[str] = None


class QueryResponse(BaseModel):
    answer: str
    confidence: Optional[float] = None
    source_questions: Optional[List[str]] = None
    intent: Optional[str] = None        # "financial_qa" | "chitchat"
    model_used: Optional[str] = None    # e.g. "gemini-1.5-flash" or "fallback"
    memory_used: bool = False           # True when long-term memory was injected
    guardrail_amended: bool = False     # True when outbound guardrail rewrote the answer
    retrieval_source: Optional[str] = None  # "faiss" | "faiss_rewritten" | "web"
    source_chunks: List[SourceChunk] = Field(default_factory=list)
    sentence_mappings: List[SentenceMapping] = Field(default_factory=list)
    status: str = "complete"  # "complete" | "awaiting_approval"
    thread_id: Optional[str] = None
    approval_card: Optional[ApprovalCard] = None
    token_usage: TokenUsageSummary = Field(default_factory=TokenUsageSummary)


class HealthResponse(BaseModel):
    status: str
    version: str = "2.0.0"


class AnalysisResponse(BaseModel):
    strategy: str
    data_type: str
    buy_signals: List[Dict[str, Any]]
    sell_signals: List[Dict[str, Any]]


class MemoryInfoResponse(BaseModel):
    user_id: str
    episode_count: int
    message: str


class MemoryClearResponse(BaseModel):
    user_id: str
    episodes_cleared: int
    message: str


class TelemetrySnapshot(BaseModel):
    started_at: str
    chat_requests: int
    cache_hits: int
    llm_calls: int
    input_tokens: int
    output_tokens: int
    total_tokens: int
    estimated_cost_usd: float
    by_model: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    by_node: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    recent_calls: List[TokenCallEvent] = Field(default_factory=list)
