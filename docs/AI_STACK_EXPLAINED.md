# AI Stack Explained — Financial Bot

> Confirms and explains every AI/ML layer in this project:
> **RAG · LangChain · LangGraph · Vector DB (FAISS)**
>
> Each section answers: what is it, where exactly is it used here, and how does it work end-to-end with a concrete example from this codebase.

---

## Quick Confirmation Table

| Technology | Used? | Where |
|---|---|---|
| RAG (Retrieval-Augmented Generation) | YES | `backend/rag/retriever.py` + `agents/nodes.py` (`rag_retrieve_node`) |
| LangChain | YES | Documents, Embeddings, VectorStore, Tools, Message types, LLM wrappers |
| LangGraph | YES | `agents/graph.py` + `agents/state.py` + `agents/nodes.py` |
| Vector DB (FAISS) | YES | `backend/rag/retriever.py` — in-memory FAISS index |

---

## 1. RAG — Retrieval-Augmented Generation

### What is RAG?

RAG is a pattern where, before calling an LLM, you first **retrieve relevant facts** from your own data store and inject them into the prompt. This grounds the LLM's answer in real knowledge rather than relying on (potentially outdated or hallucinated) training data.

The flow is:
```
User question
    │
    ▼
Search your knowledge base  ←── vector similarity
    │
    ▼
Inject top-k results as context into LLM prompt
    │
    ▼
LLM generates a grounded answer
```

### How it is implemented here

The project keeps a financial Q&A knowledge base in `backend/sample_qa.json` (20 curated Q&A pairs). RAG is split across two nodes in the LangGraph pipeline.

**Step 1 — Retrieval (`rag_retrieve_node` in `agents/nodes.py:248`)**

```python
def rag_retrieve_node(state: AgentState) -> dict:
    query = state["query"]
    docs = financial_retriever.retrieve(query, k=settings.max_results)

    context_blocks = [
        f"[{d.get('category', 'general').upper()}]\nQ: {d['question']}\nA: {d['answer']}"
        for d in docs
    ]
    context = "\n\n".join(context_blocks)

    return {
        "retrieved_docs": docs,
        "context": context,
        "source_questions": [d["question"] for d in docs],
    }
```

This node calls the retriever and formats the results into a readable context string, along with `source_questions` so the frontend can show the user *why* these results were surfaced.

**Step 2 — Generation (`gemini_generate_node` in `agents/nodes.py:283`)**

```python
user_parts = []

if summary:
    user_parts.append(f"Conversation Summary (for context):\n{summary}")

if context:
    user_parts.append(f"Knowledge Base Context:\n{context}")

user_parts.append(f"Question: {query}")
user_parts.append("Please provide an accurate and helpful answer.")

user_content = "\n\n".join(user_parts)
response = power_llm.invoke([SystemMessage(...), HumanMessage(content=user_content)])
```

The retrieved context is injected directly into the prompt. Gemini Pro sees the knowledge-base Q&A snippets and uses them as its primary source, reducing hallucination.

### Concrete example

**User asks:** *"What is a mutual fund?"*

1. The query goes to `rag_retrieve_node`.
2. FAISS finds the 3 most similar entries in `sample_qa.json` — e.g. entries about mutual funds, diversification, and NAV.
3. Those 3 Q&A pairs are formatted into `context`.
4. `gemini_generate_node` builds this prompt:

```
Knowledge Base Context:
[INVESTMENT]
Q: What is a mutual fund?
A: A mutual fund pools money from many investors to purchase a diversified portfolio...

[INVESTMENT]
Q: What does NAV mean in mutual funds?
A: Net Asset Value (NAV) is the per-share value of the fund...

Question: What is a mutual fund?
Please provide an accurate and helpful answer.
```

5. Gemini Pro returns a grounded, factual answer citing the injected context.

---

## 2. LangChain

### What is LangChain?

LangChain is a framework that provides composable building blocks for LLM applications: standard interfaces for LLMs, embeddings, vector stores, document handling, and tools. It lets you swap models (Gemini → OpenAI) without rewriting logic.

### How it is used here

LangChain is the foundational layer underneath everything. Here are its four concrete roles in this project.

---

#### 2a. Document and Embedding Model (`rag/retriever.py:13–73`)

LangChain's `Document` object is used to store each Q&A pair with structured metadata:

```python
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

# Each Q&A becomes a Document
doc = Document(
    page_content=f"Q: {item['question']}\nA: {item['answer']}",
    metadata={
        "question": item["question"],
        "answer": item["answer"],
        "category": item.get("category", "general"),
    },
)
```

`page_content` is what gets embedded (vectorised). `metadata` is stored alongside and returned at retrieval time — this is how the node can extract the original question and answer back from a search result.

The embedding model is loaded via LangChain's HuggingFace integration:

```python
self._embeddings = HuggingFaceEmbeddings(model_name=settings.embedding_model)
# model_name = "all-MiniLM-L6-v2" by default
```

This downloads `all-MiniLM-L6-v2` from HuggingFace Hub on first run and converts every `page_content` string into a 384-dimensional float vector.

---

#### 2b. LLM Wrappers — `ChatGoogleGenerativeAI` (`agents/nodes.py:32–51`)

LangChain wraps the Gemini API so it can be called with the standard `.invoke(messages)` interface:

```python
from langchain_google_genai import ChatGoogleGenerativeAI

def _build_cheap_llm():
    return ChatGoogleGenerativeAI(
        model=settings.cheap_model,       # gemini-1.5-flash
        temperature=0.0,
        google_api_key=settings.gemini_api_key,
    )

def _build_power_llm():
    return ChatGoogleGenerativeAI(
        model=settings.power_model,       # gemini-1.5-pro
        temperature=settings.gemini_temperature,
        google_api_key=settings.gemini_api_key,
    )
```

Two separate model tiers are used deliberately:
- **Flash** (cheap, fast) → used by `gatekeeper_node` and `summarize_history_node` where precision matters less
- **Pro** (capable, slower) → used only by `gemini_generate_node` for the final answer where quality matters most

This is a cost-optimisation pattern: only the expensive model is called once per conversation turn, and only when needed.

---

#### 2c. Message Types (`agents/nodes.py:21`, `agents/state.py:14`)

LangChain's standardised message classes are used throughout:

```python
from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage, SystemMessage
```

| Class | Used for |
|---|---|
| `HumanMessage` | Wrapping the user's query in prompts |
| `SystemMessage` | System/instruction prefix in every LLM call |
| `AIMessage` | Storing Gemini's response back into conversation history |
| `RemoveMessage` | Deleting old messages during history pruning (`summarize_history_node`) |

The `AgentState` stores the full conversation as `List[BaseMessage]` with LangChain's `add_messages` reducer, which handles both appending and selective deletion atomically.

---

#### 2d. Tools (`tools/financial_tools.py`)

LangChain's `@tool` decorator converts plain Python functions into structured tool definitions that any LangChain-compatible LLM can invoke via function-calling:

```python
from langchain_core.tools import tool

@tool
def search_financial_knowledge(query: str) -> str:
    """Search the internal financial knowledge base for relevant Q&A context.
    Use this tool when answering questions about stocks, investments, trading...
    """
    results = financial_retriever.retrieve(query)
    # format and return
```

The docstring becomes the tool description that the LLM reads to decide when to call it. `FINANCIAL_TOOLS = [search_financial_knowledge, get_available_trading_strategies]` exports the list for use when building a tool-calling agent variant.

---

## 3. LangGraph

### What is LangGraph?

LangGraph is a library built on top of LangChain that lets you model an LLM application as a **directed graph** of nodes and edges. Each node is a function; edges define the flow. Conditional edges allow branching based on the agent's intermediate decisions. A shared `State` object flows through every node and is updated incrementally.

This is the difference from a simple chain: a chain is a fixed A→B→C sequence. A graph can branch, loop, and route dynamically.

### How it is implemented here

**The State (`agents/state.py`)**

```python
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    query:          str
    intent:         Optional[str]       # set by router_node
    domain_status:  Optional[str]       # set by gatekeeper_node
    summary:        Optional[str]       # set by summarize_history_node
    retrieved_docs: Optional[List[dict]]# set by rag_retrieve_node
    context:        Optional[str]       # set by rag_retrieve_node
    answer:         Optional[str]       # set by generate/chitchat/out_of_scope nodes
    confidence:     Optional[float]
    source_questions: Optional[List[str]]
    error:          Optional[str]
```

`AgentState` is the single shared object. Every node receives it and returns a **partial dict** — only the fields it changes. LangGraph merges those back into the state automatically, so no node needs to pass values explicitly to the next.

**The Graph (`agents/graph.py`)**

```python
graph = StateGraph(AgentState)

# Register nodes
graph.add_node("summarize_history", summarize_history_node)
graph.add_node("gatekeeper",        gatekeeper_node)
graph.add_node("out_of_scope",      out_of_scope_node)
graph.add_node("router",            router_node)
graph.add_node("rag_retrieve",      rag_retrieve_node)
graph.add_node("gemini_generate",   gemini_generate_node)
graph.add_node("chitchat",          chitchat_node)

# Fixed edges
graph.add_edge(START,               "summarize_history")
graph.add_edge("summarize_history", "gatekeeper")
graph.add_edge("rag_retrieve",      "gemini_generate")
graph.add_edge("gemini_generate",   END)
graph.add_edge("chitchat",          END)
graph.add_edge("out_of_scope",      END)

# Conditional edges — branching based on state values
graph.add_conditional_edges("gatekeeper", _dispatch_domain, {
    "authorized":   "router",
    "unauthorized": "out_of_scope",
})
graph.add_conditional_edges("router", _dispatch_intent, {
    "financial_qa": "rag_retrieve",
    "chitchat":     "chitchat",
})

financial_agent = graph.compile()
```

**The full execution flow visualised:**

```
START
  │
  ▼
[summarize_history]
  If len(messages) > 5:
    ├── Flash LLM compresses old messages into a summary string
    └── RemoveMessage drops the old messages from state
  │
  ▼
[gatekeeper]
  Flash LLM classifies query: finance-related?
  ├── "authorized"   ──► [router]
  └── "unauthorized" ──► [out_of_scope] ──► END
                              │
                              ▼
                    "I'm only specialised in finance..."
  │
  ▼
[router]
  Keyword check: is this small-talk?
  ├── "financial_qa" ──► [rag_retrieve]
  └── "chitchat"     ──► [chitchat] ──► END
                              │
                              ▼
                    Canned reply for "hi", "thanks", etc.
  │
  ▼
[rag_retrieve]
  FAISS MMR search → top-3 Q&A pairs → state.context
  │
  ▼
[gemini_generate]
  Pro LLM prompt: summary + context + query → answer
  │
  ▼
 END
```

**Conditional edge resolvers** read a single field from state and return the next node name:

```python
def _dispatch_domain(state: AgentState) -> str:
    return state.get("domain_status", "authorized")  # written by gatekeeper_node

def _dispatch_intent(state: AgentState) -> str:
    return state.get("intent", "financial_qa")        # written by router_node
```

### Concrete example

**User asks:** *"What's the weather like today?"*

| Step | Node | What happens |
|---|---|---|
| 1 | `summarize_history` | Only 1 message — passes through, returns `{}` |
| 2 | `gatekeeper` | Flash model returns `"unauthorized"` (weather ≠ finance) |
| 3 | `out_of_scope` | Returns static refusal message |
| — | END | Graph stops; `state.answer` = refusal |

**User asks:** *"Explain RSI to me"* (6th message in a long conversation)

| Step | Node | What happens |
|---|---|---|
| 1 | `summarize_history` | 6 > 5 threshold → Flash summarises old messages, `RemoveMessage` prunes them, `state.summary` updated |
| 2 | `gatekeeper` | Returns `"authorized"` |
| 3 | `router` | Not chitchat → `"financial_qa"` |
| 4 | `rag_retrieve` | FAISS finds 3 RSI-related Q&A entries, writes `state.context` |
| 5 | `gemini_generate` | Builds prompt from `state.summary` + `state.context` + query, calls Pro model |
| — | END | `state.answer` contains a grounded RSI explanation |

---

## 4. Vector DB — FAISS

### What is a Vector Database?

A vector database stores data as high-dimensional float vectors (embeddings). Queries are also converted to vectors, and the database returns the stored items whose vectors are closest to the query vector — measured by cosine similarity or L2 distance. This enables **semantic search**: finding content that *means* the same thing, not just sharing the same keywords.

FAISS (Facebook AI Similarity Search) is an in-memory vector library that this project uses directly without a separate database server.

### How it is implemented here

**Building the index (`rag/retriever.py:45–73`)**

On application startup, `main.py` calls `retriever.initialize(qa_data)`:

```python
# 1. Load the embedding model
self._embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# 2. Build Document objects
documents = [
    Document(
        page_content=f"Q: {item['question']}\nA: {item['answer']}",
        metadata={"question": ..., "answer": ..., "category": ...}
    )
    for item in qa_data
]

# 3. Embed all documents and build the FAISS index in one call
self._vectorstore = FAISS.from_documents(documents, self._embeddings)
```

`FAISS.from_documents` does three things internally:
1. Calls `self._embeddings.embed_documents(...)` to get a 384-dim vector for each document's `page_content`
2. Builds a FAISS flat index (exact nearest-neighbour search)
3. Stores the `(vector, document)` pairs in memory

With 20 documents this index is tiny (~30 KB). For production use with thousands of documents, you would switch to `FAISS.load_local` / `FAISS.save_local` for persistence, or swap for a hosted vector DB (Pinecone, Weaviate, etc.) using the same LangChain interface.

**Searching the index — MMR (`rag/retriever.py:99–105`)**

```python
docs = self._vectorstore.max_marginal_relevance_search(
    query, k=k, fetch_k=min(k * 4, 20)
)
```

Plain similarity search returns the `k` most similar documents. The problem: if the knowledge base has 5 entries about "mutual funds", all 5 might be returned, and they carry redundant information.

**MMR (Maximal Marginal Relevance)** solves this:
1. Fetch a larger candidate set (`fetch_k = k * 4 = 12` candidates)
2. From those candidates, greedily select the next result that is both:
   - highly relevant to the query
   - dissimilar to already-selected results

This ensures the 3 returned documents cover *different aspects* of the topic, giving Gemini richer context.

If MMR fails (library version issue, etc.), the code falls back to plain similarity search:

```python
except Exception:
    docs = self._vectorstore.similarity_search(query, k=k)
```

### Concrete example

**Query:** *"How do I reduce investment risk?"*

1. The query string `"How do I reduce investment risk?"` is passed to `all-MiniLM-L6-v2`, which returns a 384-dimensional vector: `[0.12, -0.34, 0.05, ...]`.

2. FAISS computes the distance between this vector and all 20 document vectors in the index.

3. The top 12 candidates (fetch_k) by similarity might be:
   - "What is diversification?" — very high similarity
   - "What is an index fund?" — high similarity
   - "What is a mutual fund?" — high similarity (also about diversification)
   - "What is portfolio rebalancing?" — medium similarity
   - ... 8 more

4. MMR then selects 3 from those 12:
   - Picks "diversification" first (highest relevance)
   - Picks "portfolio rebalancing" next (relevant AND different from diversification)
   - Picks "index fund" next (relevant AND different from both above)
   - Skips "mutual fund" (too similar to diversification — redundant)

5. The 3 selected entries become `state.context`, injected into the Gemini prompt.

---

## How All Four Technologies Work Together

Here is the complete request lifecycle showing where each technology contributes:

```
HTTP POST /chat  {"query": "What is portfolio rebalancing?"}
        │
        ▼
   main.py
   Appends HumanMessage to state.messages          ← LangChain (message type)
        │
        ▼
   LangGraph: financial_agent.invoke(state)         ← LangGraph (graph execution)
        │
        ├─[summarize_history]
        │    RemoveMessage / summarise via Flash     ← LangChain (LLM wrapper)
        │
        ├─[gatekeeper]
        │    Flash classifies: "authorized"          ← LangChain (LLM wrapper)
        │
        ├─[router]
        │    Keyword check: "financial_qa"
        │
        ├─[rag_retrieve]
        │    Query → vector via HuggingFace model    ← LangChain (Embeddings)
        │    Vector search via FAISS MMR             ← Vector DB (FAISS)
        │    Returns top-3 Q&A docs                  ← RAG (retrieval step)
        │
        └─[gemini_generate]
             context injected into prompt            ← RAG (augmentation step)
             Gemini Pro generates grounded answer    ← LangChain (LLM wrapper)
        │
        ▼
   JSON response: {answer, source_questions, model_used}
```

Each technology is essential:
- **FAISS** makes semantic search possible (keyword search would miss paraphrased questions)
- **LangChain** provides the unified interface so the same code works with any LLM or embedding model
- **RAG** grounds Gemini's answer in curated knowledge, preventing hallucination
- **LangGraph** orchestrates the multi-step logic (guard → route → retrieve → generate) with clean separation of concerns and conditional branching
