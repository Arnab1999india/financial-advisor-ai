# Architecture Overview

## System Topology

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  CLIENT LAYER                                                               │
│   ┌──────────────────┐      ┌──────────────────────────────────────────┐   │
│   │  HTML/JS Chat UI │      │  Streamlit Dashboard (frontend/app.py)  │   │
│   └────────┬─────────┘      └──────────────────┬───────────────────────┘   │
└────────────┼──────────────────────────────────┼───────────────────────────┘
             │  HTTP/REST                        │  HTTP/REST
             ▼                                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  API LAYER  (FastAPI + Uvicorn — backend/main.py)                           │
│                                                                             │
│   GET  /health          POST /chat            POST /analyze                 │
│   GET  /info            (QueryRequest)        (UploadFile + Form)           │
└──────────────────────────────┬──────────────────────────┬───────────────────┘
                               │                          │
                               ▼                          ▼
┌──────────────────────────────────────┐  ┌──────────────────────────────────┐
│  LANGGRAPH AGENT LAYER               │  │  ANALYSIS ENGINE                 │
│  (backend/agents/)                   │  │  (backend/analysis.py +          │
│                                      │  │   backend/strategies.py)         │
│  ┌──────────┐                        │  │                                  │
│  │  router  │ ── chitchat ──────┐    │  │  load_data_from_file()           │
│  └────┬─────┘                   │    │  │  run_analysis()                  │
│       │ financial_qa            ▼    │  │                                  │
│       ▼                   ┌─────────┐│  │  Strategies:                     │
│  ┌────────────┐            │chitchat ││  │  • swing_trading                 │
│  │rag_retrieve│            │  node   ││  │  • scalping_trading              │
│  └─────┬──────┘            └─────────┘│  │  • 44_moving_average             │
│        │                             │  │  • trend_trading                 │
│        ▼                             │  │  • range_trading                 │
│  ┌─────────────────┐                 │  └──────────────────────────────────┘
│  │gemini_generate  │                 │
│  │     node        │                 │
│  └─────────────────┘                 │
└──────────────────────────────────────┘
         │                   │
         ▼                   ▼
┌─────────────────┐  ┌────────────────────────────────────┐
│  RAG LAYER      │  │  LLM LAYER                         │
│  (backend/rag/) │  │  Google Gemini (via LangChain)     │
│                 │  │                                    │
│ FAISS VectorDB  │  │  ChatGoogleGenerativeAI            │
│ HuggingFace     │  │  Model: gemini-1.5-flash (default) │
│ Embeddings      │  │  Temperature: 0.2                  │
│ (MMR Search)    │  │                                    │
└─────────────────┘  └────────────────────────────────────┘
```

---

## LangGraph Agent Workflow

```
              ┌─────────────────────────────┐
              │        User Query           │
              └──────────────┬──────────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │   router_node   │  ← lightweight keyword classifier
                    └────────┬────────┘
                             │
              ┌──────────────┴──────────────┐
              │                             │
     intent="chitchat"           intent="financial_qa"
              │                             │
              ▼                             ▼
    ┌──────────────────┐       ┌─────────────────────────┐
    │   chitchat_node  │       │    rag_retrieve_node     │
    │  (static reply)  │       │  FAISS MMR Search (k=3) │
    └────────┬─────────┘       └────────────┬────────────┘
             │                              │
             │                    retrieved_docs, context
             │                              │
             │                              ▼
             │               ┌──────────────────────────────┐
             │               │     gemini_generate_node      │
             │               │  • If API key set → Gemini   │
             │               │  • Else → best-match fallback │
             │               └──────────────┬───────────────┘
             │                              │
             └──────────────┬───────────────┘
                            │
                            ▼
                   ┌─────────────────┐
                   │  QueryResponse  │
                   │  answer         │
                   │  confidence     │
                   │  source_qs      │
                   │  intent         │
                   │  model_used     │
                   └─────────────────┘
```

---

## RAG Pipeline

```
Sample Q&A JSON
      │
      ▼  (startup)
HuggingFace Embeddings
(all-MiniLM-L6-v2)
      │
      ▼
FAISS VectorStore
(Document objects with metadata:
 question, answer, category)
      │
      │  (per request)
      ▼
MMR Search (k=max_results)
      │
      ▼
Context Builder
(formatted Q&A blocks with category labels)
      │
      ▼
Gemini prompt injection
```

---

## Technology Stack

| Layer          | Technology                          | Version         |
|----------------|-------------------------------------|-----------------|
| Web Framework  | FastAPI                             | latest          |
| ASGI Server    | Uvicorn                             | latest          |
| Agent Graph    | LangGraph                           | latest          |
| LLM            | Google Gemini (langchain-google-genai) | gemini-1.5-flash |
| Embeddings     | HuggingFace sentence-transformers   | all-MiniLM-L6-v2 |
| Vector Store   | FAISS                               | faiss-cpu       |
| LangChain Core | langchain, langchain-core, langchain-community | latest |
| Data           | pandas, numpy, pandas-ta            | see requirements |
| Validation     | Pydantic v2, pydantic-settings      | latest          |
| Frontend       | HTML/CSS/JS + Streamlit (optional)  | -               |
| Testing        | pytest + httpx                      | latest          |
