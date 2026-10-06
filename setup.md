# Setup Guide

## Prerequisites

- Python 3.10 or higher
- Java (required only if you use PDF files in the `/analyze` endpoint — used by `tabula-py`)
- A free [Gemini API key](https://aistudio.google.com/app/apikey)

---

## 1. Clone the Repository

```bash
git clone <repo-url>
cd financial_bot
```

---

## 2. Backend Setup

```bash
cd backend

# Create and activate a virtual environment
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Configure Environment Variables

```bash
cp .env.example .env
```

Open `.env` and set your Gemini API key:

```env
GEMINI_API_KEY=your-gemini-api-key-here
```

All available variables (the rest have sensible defaults):

| Variable | Default | Description |
|---|---|---|
| `GEMINI_API_KEY` | *(required)* | Gemini LLM key — get one at [aistudio.google.com](https://aistudio.google.com/app/apikey) |
| `GEMINI_MODEL` | `gemini-1.5-flash` | Model variant (`gemini-1.5-flash`, `gemini-1.5-pro`, `gemini-2.0-flash`) |
| `GEMINI_TEMPERATURE` | `0.2` | Response creativity — `0` is deterministic, `1` is creative |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | HuggingFace sentence-transformer for RAG |
| `MAX_RESULTS` | `3` | Number of RAG documents retrieved per query |
| `DATA_FILE` | `sample_qa.json` | Knowledge base file (relative to `backend/`) |
| `DEBUG` | `false` | Enable debug logging |

> **Fallback mode:** If `GEMINI_API_KEY` is missing or left as the placeholder, the app runs without an LLM and returns the best-matched RAG result directly.

### Start the Backend

```bash
uvicorn main:app --reload --port 8000
```

- API base URL: `http://localhost:8000`
- Interactive docs (Swagger): `http://localhost:8000/docs`

---

## 3. Frontend Setup

You have two options — run either or both.

### Option A — Streamlit Dashboard (recommended)

```bash
cd frontend
pip install streamlit
streamlit run app.py
```

Opens at `http://localhost:8501`. Provides a two-tab UI:
- **Chat** — conversational financial Q&A
- **Trading Signals** — upload a CSV/XLS/XLSX file and pick a strategy

### Option B — Static HTML Chat

```bash
cd frontend
python -m http.server 3000
```

Opens at `http://localhost:3000`. Lightweight single-page chat interface.

---

## 4. API Endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/` | Health check |
| `GET` | `/health` | Detailed system status |
| `GET` | `/info` | Version, model info, RAG stats |
| `POST` | `/chat` | Financial Q&A — body: `{"query": "..."}` |
| `POST` | `/analyze` | Trading signals — multipart: `file`, `strategy`, `data_type` |

### Supported Trading Strategies

Pass one of these as `strategy` in the `/analyze` request:

| Value | Indicator |
|---|---|
| `swing_trading` | SMA 20/50 crossover |
| `scalping_trading` | RSI (14) |
| `44_moving_average` | SMA-44 vs price |
| `trend_trading` | MACD crossover |
| `range_trading` | Bollinger Bands |

### Data File Format

Your uploaded file must have these columns: `Date`, `Open`, `High`, `Low`, `Close`.  
Accepted formats: `.csv`, `.xls`, `.xlsx`, `.pdf` (PDF requires Java).  
Minimum **30 rows** of data required.

---

## 5. Running Tests

```bash
cd backend
pytest tests/
```

---

## 6. Project Structure

```
financial_bot/
├── backend/
│   ├── agents/          # LangGraph agent graph, nodes, and state
│   ├── rag/             # FAISS vector store + MMR retrieval
│   ├── tools/           # LangChain tool definitions
│   ├── tests/           # Pytest test suite
│   ├── main.py          # FastAPI app (entry point)
│   ├── config.py        # Pydantic settings (reads .env)
│   ├── analysis.py      # File upload + data cleaning
│   ├── strategies.py    # Technical analysis strategy implementations
│   ├── sample_qa.json   # Built-in financial knowledge base (20 Q&A pairs)
│   ├── requirements.txt
│   └── .env.example
├── frontend/
│   ├── app.py           # Streamlit dashboard
│   ├── index.html       # Static chat interface
│   ├── script.js
│   └── styles.css
├── docs/                # HLD, LLD, API contracts, architecture, deployment
└── setup.md             # This file
```

---

## 7. Troubleshooting

**Backend won't start — missing module**
```bash
pip install -r backend/requirements.txt
```

**PDF analysis fails**
Install Java (JDK 11+) and make sure `java` is on your `PATH`. `tabula-py` requires it to parse PDF tables.

**LLM not responding / fallback mode active**
Check that `GEMINI_API_KEY` in `backend/.env` is set to a real key and not the placeholder string.

**Streamlit can't reach the backend**
Ensure the backend is running on port `8000`. The Streamlit app connects to `http://127.0.0.1:8000` by default (configured in `frontend/app.py` line 6).

**`sentence-transformers` download is slow**
The embedding model (`all-MiniLM-L6-v2`) is downloaded from HuggingFace on first run (~90 MB). Subsequent starts use the local cache.
