# API Contracts

**Base URL (local):** `http://localhost:8000`
**Interactive Docs:** `http://localhost:8000/docs` (Swagger UI)
**ReDoc:** `http://localhost:8000/redoc`

---

## Endpoints

### `GET /`

Health check (root).

**Response `200 OK`**
```json
{
  "status": "healthy",
  "version": "2.0.0"
}
```

---

### `GET /health`

Detailed health status.

**Response `200 OK`**
```json
{
  "status": "healthy",
  "version": "2.0.0"
}
```

`status` is `"initializing"` if the FAISS index is not yet ready.

---

### `GET /info`

Runtime system metadata.

**Response `200 OK`**
```json
{
  "version": "2.0.0",
  "app_name": "Financial Bot API",
  "rag_ready": true,
  "total_qa_pairs": 20,
  "embedding_model": "all-MiniLM-L6-v2",
  "llm_model": "gemini-1.5-flash",
  "data_file": "sample_qa.json",
  "max_rag_results": 3
}
```

---

### `POST /chat`

Ask a financial question. Routes through the LangGraph agent (router → RAG → Gemini).

**Request**
```
Content-Type: application/json
```
```json
{
  "query": "What is the difference between a stock and a bond?",
  "session_id": null
}
```

| Field | Type | Required | Notes |
|-------|------|----------|-------|
| `query` | `string` | Yes | Natural-language question |
| `session_id` | `string \| null` | No | Reserved; currently unused |

**Response `200 OK`**
```json
{
  "answer": "A stock represents an ownership stake in a company, while a bond is a debt instrument...",
  "confidence": null,
  "source_questions": [
    "What is a bond?",
    "What is the stock market?"
  ],
  "intent": "financial_qa",
  "model_used": "gemini-1.5-flash"
}
```

| Field | Type | Notes |
|-------|------|-------|
| `answer` | `string` | Generated or retrieved answer |
| `confidence` | `float \| null` | 0–1; populated only on fallback paths |
| `source_questions` | `string[] \| null` | RAG context sources shown to user |
| `intent` | `string \| null` | `"financial_qa"` or `"chitchat"` |
| `model_used` | `string \| null` | `"gemini-1.5-flash"`, `"fallback"`, or `"static"` |

**Response `503 Service Unavailable`**
```json
{ "detail": "Service initializing. Please retry shortly." }
```

**Response `500 Internal Server Error`**
```json
{ "detail": "Internal server error" }
```

---

### `POST /analyze`

Upload OHLC data and receive algorithmic trading signals.

**Request**
```
Content-Type: multipart/form-data
```

| Field | Type | Required | Notes |
|-------|------|----------|-------|
| `file` | `File` | Yes | CSV, XLS, XLSX, or PDF containing OHLC columns |
| `strategy` | `string` | Yes | One of the strategy keys listed below |
| `data_type` | `string` | Yes | `"swing"` or `"intraday"` |

**Required CSV columns:** `Date`, `Open`, `High`, `Low`, `Close` (case-insensitive)

**Available strategy keys:**

| Key | Description |
|-----|-------------|
| `swing_trading` | SMA-20/50 Moving Average Crossover |
| `scalping_trading` | RSI-based (oversold/overbought) |
| `44_moving_average` | Price vs 44-period SMA |
| `trend_trading` | MACD Crossover |
| `range_trading` | Bollinger Bands |

**Response `200 OK`**
```json
{
  "strategy": "swing_trading",
  "data_type": "swing",
  "buy_signals": [
    {
      "Date": "2024-01-15T00:00:00",
      "Open": 150.25,
      "High": 153.80,
      "Low": 149.10,
      "Close": 152.40,
      "short_ma": 148.5,
      "long_ma": 145.2,
      "buy_signal": true,
      "sell_signal": false
    }
  ],
  "sell_signals": [
    {
      "Date": "2024-02-10T00:00:00",
      "Open": 160.00,
      "High": 162.50,
      "Low": 158.75,
      "Close": 159.30,
      "short_ma": 158.0,
      "long_ma": 160.5,
      "buy_signal": false,
      "sell_signal": true
    }
  ]
}
```

**Response `400 Bad Request`**
```json
{ "detail": "Input data must contain the following columns: ['Date', 'Open', 'High', 'Low', 'Close']" }
```

Other `400` causes: unsupported file format, invalid strategy name, no tables in PDF.

**Response `500 Internal Server Error`**
```json
{ "detail": "Internal server error" }
```

---

## Error Code Summary

| HTTP Status | When |
|-------------|------|
| `200 OK` | Successful response |
| `400 Bad Request` | Invalid input (bad strategy, missing columns, unsupported file) |
| `422 Unprocessable Entity` | Pydantic validation failure (missing required fields, wrong types) |
| `503 Service Unavailable` | FAISS index not yet initialised at startup |
| `500 Internal Server Error` | Unexpected runtime error |

---

## Example cURL Calls

```bash
# Health check
curl http://localhost:8000/health

# Chat
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "What is dollar-cost averaging?"}'

# Trading analysis
curl -X POST http://localhost:8000/analyze \
  -F "file=@prices.csv" \
  -F "strategy=swing_trading" \
  -F "data_type=swing"

# System info
curl http://localhost:8000/info
```

---

## OpenAPI Schema

The full auto-generated OpenAPI 3.1 schema is available at:
- JSON: `GET /openapi.json`
- Swagger UI: `GET /docs`
- ReDoc: `GET /redoc`
