# Low-Level Design (LLD)

## 1. Module Map

```
backend/
├── main.py                  # FastAPI app + lifecycle hooks
├── config.py                # Pydantic-Settings configuration
├── models.py                # Pydantic request/response schemas
├── logger.py                # Logging setup (stdout, INFO level)
├── analysis.py              # File ingestion + DataFrame cleaning
├── strategies.py            # Technical indicator strategies
├── sample_qa.json           # Curated financial Q&A knowledge base
│
├── agents/
│   ├── state.py             # AgentState TypedDict (LangGraph)
│   ├── nodes.py             # Node functions: router, rag_retrieve, gemini_generate, chitchat
│   └── graph.py             # StateGraph definition + compilation
│
├── rag/
│   └── retriever.py         # FinancialRetriever (FAISS + HuggingFace embeddings)
│
└── tools/
    └── financial_tools.py   # LangChain @tool functions
```

---

## 2. Configuration (`config.py`)

```python
class Settings(BaseSettings):
    app_name: str              = "Financial Bot API"
    debug: bool                = False
    embedding_model: str       = "all-MiniLM-L6-v2"
    max_results: int           = 3
    data_file: str             = "sample_qa.json"
    gemini_api_key: str | None = None          # GEMINI_API_KEY env var
    gemini_model: str          = "gemini-1.5-flash"
    gemini_temperature: float  = 0.2
```

**Override precedence**: env var > `.env` file > class default.

---

## 3. Data Models (`models.py`)

### QueryRequest
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `query` | `str` | Yes | User's natural-language question |
| `session_id` | `str \| None` | No | Reserved for future multi-turn support |

### QueryResponse
| Field | Type | Description |
|-------|------|-------------|
| `answer` | `str` | Generated or retrieved answer |
| `confidence` | `float \| None` | 0–1 score (fallback paths only; Gemini returns `null`) |
| `source_questions` | `list[str] \| None` | Q&A questions that informed the answer |
| `intent` | `str \| None` | `"financial_qa"` or `"chitchat"` |
| `model_used` | `str \| None` | `"gemini-1.5-flash"`, `"fallback"`, or `"static"` |

### AnalysisResponse
| Field | Type | Description |
|-------|------|-------------|
| `strategy` | `str` | Strategy name used |
| `data_type` | `str` | Data type hint passed in request |
| `buy_signals` | `list[dict]` | Rows where `buy_signal == True` (OHLC + indicators) |
| `sell_signals` | `list[dict]` | Rows where `sell_signal == True` |

---

## 4. LangGraph Agent

### 4.1 State (`agents/state.py`)

```python
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    query: str
    intent: Optional[str]              # "financial_qa" | "chitchat"
    retrieved_docs: Optional[List[dict]]
    context: Optional[str]             # formatted context for LLM
    answer: Optional[str]
    confidence: Optional[float]
    source_questions: Optional[List[str]]
    error: Optional[str]
```

`messages` uses `operator.add` so LangGraph *appends* rather than replaces.

### 4.2 Nodes (`agents/nodes.py`)

#### `router_node(state) → dict`
- Checks if query is short chitchat (< 30 chars + keyword match).
- Sets `intent = "chitchat"` or `"financial_qa"`.
- **Never** calls an external API (fast, deterministic).

#### `rag_retrieve_node(state) → dict`
- Calls `financial_retriever.retrieve(query, k=max_results)`.
- Builds `context` string from retrieved Q&A blocks with category labels.
- Sets `retrieved_docs`, `context`, `source_questions`.
- On error: sets empty results + `error` field; does not raise.

#### `gemini_generate_node(state) → dict`
- If `GEMINI_API_KEY` not set → fallback to `retrieved_docs[0].answer`.
- If key set → instantiates `ChatGoogleGenerativeAI` and invokes with:
  - `SystemMessage` (grounding instructions)
  - `HumanMessage` (query + context)
- On LLM error → graceful fallback with `confidence=0.6`.
- Sets `answer`, `confidence`, `model_used`.

#### `chitchat_node(state) → dict`
- Pure dict lookup; no I/O.
- Sets `answer` from static reply map, `confidence=1.0`.

### 4.3 Graph (`agents/graph.py`)

```
START → router_node
router_node ──[financial_qa]──► rag_retrieve_node → gemini_generate_node → END
router_node ──[chitchat]──────► chitchat_node → END
```

Compiled as a singleton `financial_agent` at module load time.

---

## 5. RAG Retriever (`rag/retriever.py`)

### Class: `FinancialRetriever`

| Method | Signature | Description |
|--------|-----------|-------------|
| `initialize` | `(qa_data: list[dict]) → None` | Builds HuggingFace embeddings + FAISS index |
| `retrieve` | `(query: str, k: int) → list[dict]` | MMR search; falls back to similarity search |
| `is_ready` | property `→ bool` | True after `initialize()` completes |

**Document format in FAISS:**
```python
Document(
    page_content = "Q: {question}\nA: {answer}",
    metadata     = {"question": ..., "answer": ..., "category": ...}
)
```

Indexing both question and answer improves recall for answer-oriented queries.

**MMR parameters:**
- `k = max_results` (default 3)
- `fetch_k = min(k × 4, 20)` — fetch wider pool before MMR reranking

---

## 6. LangChain Tools (`tools/financial_tools.py`)

| Tool | Input | Output |
|------|-------|--------|
| `search_financial_knowledge` | `query: str` | Formatted context string from FAISS |
| `get_available_trading_strategies` | *(none)* | JSON string of strategy descriptions |

Tools are OpenAI-function-calling-compatible through LangChain's `@tool` decorator, meaning they can be bound to any supported LLM using `.bind_tools(FINANCIAL_TOOLS)`.

---

## 7. Analysis Engine

### `load_data_from_file(file: UploadFile) → pd.DataFrame`

1. Detect extension: `csv` | `xls` | `xlsx` | `pdf`
2. Parse file → raw DataFrame
3. Standardise column names (`.capitalize()`)
4. Validate required columns: `Date`, `Open`, `High`, `Low`, `Close`
5. Coerce OHLC to numeric; drop rows with NaN
6. Parse `Date` column to `datetime`; drop unparseable rows

### Strategy Functions (in `strategies.py`)

| Key | Indicators | Buy Condition | Sell Condition |
|-----|-----------|---------------|----------------|
| `swing_trading` | SMA-20, SMA-50 | Short MA crosses above Long MA | Short MA crosses below Long MA |
| `scalping_trading` | RSI-14 | RSI < 30 | RSI > 70 |
| `44_moving_average` | SMA-44 | Close > SMA-44 | Close < SMA-44 |
| `trend_trading` | MACD-12-26-9 | MACD crosses above Signal | MACD crosses below Signal |
| `range_trading` | Bollinger Bands-20 | Close ≤ Lower Band | Close ≥ Upper Band |

---

## 8. Error Handling Strategy

| Layer | Error Type | Handling |
|-------|-----------|----------|
| FastAPI endpoints | `ValueError` | 400 Bad Request |
| FastAPI endpoints | Unhandled `Exception` | 500 Internal Server Error |
| `rag_retrieve_node` | Any exception | Sets `error` field; returns empty docs; does not crash graph |
| `gemini_generate_node` | No API key | Fallback to retrieved doc |
| `gemini_generate_node` | LLM exception | Fallback to retrieved doc + `confidence=0.6` |
| Startup | Critical failure | Re-raises (app fails to start with clear log message) |

---

## 9. Logging

- Module: `logger.py` → `logging.getLogger("financial_bot")`
- Level: `INFO` globally; `DEBUG` available via `settings.debug`
- Format: `%(asctime)s - %(name)s - %(levelname)s - %(message)s`
- Handler: `StreamHandler(sys.stdout)` (container-friendly)
- Key log points: startup steps, query routing, retrieval count, LLM call outcome, errors
