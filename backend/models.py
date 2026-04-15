from pydantic import BaseModel
from typing import List, Optional, Dict, Any


class QueryRequest(BaseModel):
    query: str
    session_id: Optional[str] = None  # Reserved for multi-turn conversation support


class QueryResponse(BaseModel):
    answer: str
    confidence: Optional[float] = None
    source_questions: Optional[List[str]] = None
    intent: Optional[str] = None        # "financial_qa" | "chitchat"
    model_used: Optional[str] = None    # e.g. "gemini-1.5-flash" or "fallback"


class HealthResponse(BaseModel):
    status: str
    version: str = "2.0.0"


class AnalysisResponse(BaseModel):
    strategy: str
    data_type: str
    buy_signals: List[Dict[str, Any]]
    sell_signals: List[Dict[str, Any]]
