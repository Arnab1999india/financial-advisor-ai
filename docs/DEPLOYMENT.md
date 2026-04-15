# Deployment Guide

## Prerequisites

| Requirement | Version |
|------------|---------|
| Python | 3.10+ |
| pip | latest |
| Java (optional) | 8+ (required for PDF parsing via tabula-py) |

---

## 1. Local Development Setup

```bash
# 1. Clone / navigate to project
cd financial_bot/backend

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate        # Linux / macOS
# venv\Scripts\activate         # Windows PowerShell
# .\venv\Scripts\Activate.ps1   # Windows (alternative)

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
# Edit .env and set GEMINI_API_KEY (get free key at https://aistudio.google.com/app/apikey)

# 5. Start the server
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

**API available at:** `http://localhost:8000`
**Swagger UI:** `http://localhost:8000/docs`

---

## 2. Environment Variables Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `DEBUG` | `false` | Enable debug mode |
| `APP_NAME` | `"Financial Bot API"` | Application display name |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | HuggingFace sentence-transformer model |
| `MAX_RESULTS` | `3` | Number of RAG documents retrieved per query |
| `DATA_FILE` | `sample_qa.json` | Path to Q&A knowledge base (relative to `backend/`) |
| `GEMINI_API_KEY` | *(unset)* | Google Gemini API key. Unset = fallback mode |
| `GEMINI_MODEL` | `gemini-1.5-flash` | Gemini model identifier |
| `GEMINI_TEMPERATURE` | `0.2` | LLM temperature (0 = deterministic, 1 = creative) |

### Fallback Mode (no API key)

If `GEMINI_API_KEY` is unset or equals `your-gemini-api-key-here`, the bot runs in **fallback mode**:
- `/chat` returns the best-matched Q&A pair from the FAISS index (no LLM call).
- `/analyze` works normally (no LLM required).
- `model_used` in the response will be `"fallback"`.

---

## 3. Production Deployment

### 3.1 Gunicorn + Uvicorn Workers

```bash
pip install gunicorn

gunicorn main:app \
  --worker-class uvicorn.workers.UvicornWorker \
  --workers 4 \
  --bind 0.0.0.0:8000 \
  --timeout 120 \
  --access-logfile -
```

Recommended workers: `2 × CPU_cores + 1`.

### 3.2 Docker

```dockerfile
# Dockerfile (place in backend/)
FROM python:3.11-slim

WORKDIR /app

# System dependency for tabula-py (PDF parsing)
RUN apt-get update && apt-get install -y default-jre-headless && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
```

```bash
# Build
docker build -t financial-bot ./backend

# Run
docker run -p 8000:8000 \
  -e GEMINI_API_KEY=your-key-here \
  -e GEMINI_MODEL=gemini-1.5-flash \
  financial-bot
```

### 3.3 docker-compose

```yaml
# docker-compose.yml
version: "3.9"
services:
  api:
    build: ./backend
    ports:
      - "8000:8000"
    environment:
      - GEMINI_API_KEY=${GEMINI_API_KEY}
      - GEMINI_MODEL=gemini-1.5-flash
      - GEMINI_TEMPERATURE=0.2
      - MAX_RESULTS=3
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
```

```bash
GEMINI_API_KEY=your-key docker-compose up -d
```

---

## 4. Frontend Setup

### Option A: Static HTML/JS Chat

```bash
cd frontend
python -m http.server 3000
# Open: http://localhost:3000
```

The frontend calls `http://localhost:8000` by default. Update the API URL in `frontend/script.js` if the backend is on a different host.

### Option B: Streamlit Dashboard

```bash
cd frontend
pip install streamlit
streamlit run app.py
# Open: http://localhost:8501
```

---

## 5. Running Tests

```bash
cd backend
pip install pytest httpx
pytest tests/ -v
```

---

## 6. Extending the Knowledge Base

Add entries to `backend/sample_qa.json`:

```json
{
  "question": "What is a hedge fund?",
  "answer": "A hedge fund is a pooled investment fund that uses advanced strategies...",
  "category": "investment_vehicles"
}
```

Restart the server — the FAISS index is rebuilt at startup.

**Category values** (for metadata filtering, future use):
`basics`, `investment_vehicles`, `trading`, `technical_analysis`, `risk_management`, `macroeconomics`, `market_concepts`, `strategies`, `valuation`

---

## 7. Changing the LLM

Switch to a different Gemini model by updating `.env`:

```
GEMINI_MODEL=gemini-1.5-pro      # More capable, slower
GEMINI_MODEL=gemini-1.5-flash    # Fast, cost-efficient (default)
GEMINI_MODEL=gemini-2.0-flash    # Latest generation
```

To switch to a different LangChain-supported LLM provider, update `agents/nodes.py`'s `_build_llm()` function and add the corresponding package to `requirements.txt`. The rest of the graph is provider-agnostic.
