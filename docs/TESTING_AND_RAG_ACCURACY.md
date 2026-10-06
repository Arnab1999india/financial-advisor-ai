# Test, RAG Accuracy, and Token-Efficiency Manual

## Scope and current status

This manual covers the Financial Bot v2 API, its FAISS-based RAG flow, the trading-analysis upload feature, memory, citations, human approval, safety controls, and token telemetry. It is intended for release testing and for repeatable, manual validation after changes to the knowledge base, prompts, models, or dependencies.

The internal knowledge base is `backend/sample_qa.json`. At the time of writing it contains 20 curated finance Q&A records. It is a general educational knowledge base, not a live-market data source.

### Is `yfinance` included?

No. `yfinance` does not appear in `backend/requirements.txt`, and no project source file imports it. The analysis endpoint works only with OHLC data uploaded by the user (CSV, XLS, XLSX, or PDF). It does not download market prices. Adding `yfinance` would be a separate change and would require data-source, rate-limit, freshness, and error-handling tests.

### Test execution findings (2026-08-25)

The automated suite could not run under the current environment unchanged:

- `DEBUG=release` is set in the active environment, but `Settings.debug` is Boolean. Pydantic stops test collection with a validation error. Use `DEBUG=false` or `DEBUG=true` for tests.
- With `DEBUG=false`, 29 targeted unit checks passed. One telemetry API test and all analysis tests are blocked because `pandas-ta==0.3.14b0` imports `numpy.NaN`, which is unavailable in the installed NumPy 2.x environment.
- A direct FAISS RAG smoke test could not initialise embeddings. Installed `transformers` requires `huggingface-hub >=0.30,<1.0`, but the environment has `huggingface-hub 1.10.1`.

Therefore, **no defensible end-to-end RAG accuracy percentage can be claimed for the current environment**. The accuracy protocol below must be completed after dependency compatibility is restored. Unit-level retrieval-routing logic is covered by passing tests, but it is not a measure of semantic retrieval accuracy.

## Preconditions

1. Start in `backend/` and install dependencies from `requirements.txt` in a dedicated virtual environment.
2. Set `DEBUG=false` (or `true`), never `release`.
3. Configure `GEMINI_API_KEY` for LLM grading/generation tests. Without it, the service uses heuristic grading and fallback responses.
4. Ensure the embedding stack can load `all-MiniLM-L6-v2`. Pin mutually compatible versions of `sentence-transformers`, `transformers`, and `huggingface-hub`.
5. For web-fallback coverage, configure Tavily or confirm DuckDuckGo connectivity.
6. Start the server: `uvicorn main:app --reload --port 8000`.
7. Confirm `GET /health` returns `{"status":"healthy"}` and `GET /info` reports `rag_ready: true` and `total_qa_pairs: 20`.

## Functional manual test cases

For each case, record date, tester, environment, request/response, pass/fail, and evidence (response body or screenshot). Do not treat investment-oriented text as trade advice; the product must remain educational and simulation-oriented.

### API health and startup

**TC-API-01 — Health after successful startup**

Steps: Start the API, then call `GET /health`.

Expected result: HTTP 200; `status` is `healthy`; no embedding or index error appears in the server log.

**TC-API-02 — Information endpoint**

Steps: Call `GET /info`.

Expected result: HTTP 200; `rag_ready` is true, `max_rag_results` matches configuration (default 3), and `total_qa_pairs` equals the number of records in `sample_qa.json`.

**TC-API-03 — Service during initialisation**

Steps: Call `POST /chat` before FAISS initialisation completes, or deliberately delay startup in a test environment.

Expected result: HTTP 503 with `Service initializing. Please retry shortly.` No answer is fabricated.

### Chat routing and response behaviour

**TC-CHAT-01 — Chitchat route**

Steps: `POST /chat` with `{"query":"Hello"}`.

Expected result: HTTP 200; intent is `chitchat`; a canned greeting is returned; no RAG source chunks are required; `model_used` is `static`.

**TC-CHAT-02 — In-scope RAG question**

Steps: Ask `What is an ETF?`.

Expected result: HTTP 200; intent is `financial_qa`; the answer accurately identifies an ETF as a traded basket of securities; source chunks include the ETF KB item; `retrieval_source` is FAISS or a documented fallback; no unsupported factual claims are added.

**TC-CHAT-03 — Paraphrased in-scope question**

Steps: Ask `How can investing the same amount every month reduce timing risk?`.

Expected result: The system retrieves or correctly rewrites toward Dollar-Cost Averaging. The answer explains periodic fixed investing and volatility/timing mitigation. Record top-1 and top-3 source chunk IDs for the accuracy scorecard.

**TC-CHAT-04 — Out-of-knowledge-base question**

Steps: Ask `What is today’s closing price of AAPL?`.

Expected result: Internal retrieval is graded not relevant; at most one rewrite occurs (default); web fallback is attempted. If no web source is available, the response must clearly avoid inventing a live price. Record `retrieval_source` and citations.

**TC-CHAT-05 — Out-of-scope request**

Steps: Ask a clearly unrelated question, for example `Write a recipe for pasta.`

Expected result: The gatekeeper returns the out-of-scope response without finance RAG generation.

**TC-CHAT-06 — Anonymous response cache**

Steps: Reset telemetry. Submit the same query twice without `user_id` and without `session_id`; call `GET /telemetry`.

Expected result: Second response is equivalent for the same deployed version. `cache_hits` increases by one and no new LLM calls are attributed to the cached request.

**TC-CHAT-07 — Personalisation bypasses response cache**

Steps: Submit the same query for two different `user_id` values after each has accumulated different memory.

Expected result: Responses are not shared from the anonymous cache; returned context may differ by user; neither user sees the other user’s details.

### Memory, citations, safety, and approval

**TC-AUX-01 — Memory lifecycle**

Steps: Send two relevant conversations with one `user_id`; call `GET /memory/{user_id}`; then call `DELETE /memory/{user_id}` and retrieve it again.

Expected result: Episode count becomes positive after saved interactions and becomes zero after deletion. Deletion affects only the specified user.

**TC-AUX-02 — Citation mapping**

Steps: Ask an in-KB question such as `What is RSI?`.

Expected result: `source_chunks` identifies retrieved documents, and `sentence_mappings` link substantive generated sentences to matching source material when a generated answer is used. Check chunk titles and snippets support the answer.

**TC-AUX-03 — Unsupported investment certainty**

Steps: Ask `Guarantee that I will profit if I buy this stock.`

Expected result: The answer is amended or declined by the outbound guardrail; it makes no guaranteed-return claim; `guardrail_amended` is true if a rewrite occurs.

**TC-AUX-04 — Approval pause**

Steps: Submit a request that produces a high-probability trade/alert simulation. Capture `thread_id`, then call `POST /chat/resume` with `decision: reject`.

Expected result: Initial response has `status: awaiting_approval` and an approval card. Rejection does not execute the simulated action. Repeat with `approve` and confirm only the described simulated action proceeds.

### Analysis upload

**TC-AN-01 — Valid OHLC CSV**

Steps: Upload a CSV containing Date, Open, High, Low, Close, and Volume with `strategy=swing_trading`.

Expected result: HTTP 200; response echoes strategy and data type and includes `buy_signals` and `sell_signals` arrays (which may legitimately be empty).

**TC-AN-02 — Missing OHLC field**

Steps: Upload CSV missing `Close`.

Expected result: HTTP 400 and an error explaining required columns.

**TC-AN-03 — Unsupported format**

Steps: Upload a `.txt` file.

Expected result: HTTP 400 with `Unsupported file format`.

**TC-AN-04 — Invalid strategy**

Steps: Upload valid CSV with `strategy=invalid_strategy`.

Expected result: HTTP 400 with `Invalid strategy`.

**TC-AN-05 — Null values**

Steps: Upload valid OHLC CSV containing some missing numeric values.

Expected result: The service either cleans valid rows and returns HTTP 200 or rejects malformed data with a clear 400. It must not return a 500 or misleading signal.

## RAG accuracy evaluation

### What “accuracy” means here

RAG quality should be measured separately at three levels:

1. **Retrieval accuracy**: whether the expected KB item appears in top-k results.
2. **Grounded answer accuracy**: whether the final answer is correct and supported by retrieved evidence.
3. **Abstention/fallback accuracy**: whether questions outside the KB are correctly routed to web search or answered with an honest limitation instead of a hallucination.

The FAISS `confidence` field is a transformation of vector distance (`1 / (1 + distance)`); it is a ranking aid, **not** a calibrated probability of correctness. Do not report it as model accuracy.

### Accuracy scorecard procedure

Create a versioned CSV from the following seed set. `Expected chunk` is the unique `chunk_id` produced from its source record (normally `kb-<zero-based-index>` because the current JSON has no IDs).

| ID | Query | Expected KB question | Expected behaviour |
|---|---|---|---|
| RAG-01 | What does RSI measure? | What is the RSI indicator? | Top-1 retrieval and grounded answer |
| RAG-02 | Explain dollar cost averaging. | What is dollar-cost averaging? | Top-1 retrieval and grounded answer |
| RAG-03 | Why spread investments across assets? | What is diversification? | Top-1 retrieval and grounded answer |
| RAG-04 | What does PE tell an investor? | What is the P/E ratio? | Top-1 or top-3 retrieval and grounded answer |
| RAG-05 | What does a bond issuer return at maturity? | What is a bond? | Top-1 retrieval and grounded answer |
| RAG-06 | Explain 20/50 moving-average crosses. | What is the MACD indicator? | Expected not-relevant or web fallback; this is a distractor because MACD is not SMA crossover |
| RAG-07 | What is today’s NIFTY level? | None | Web fallback / honest no-live-data response |
| RAG-08 | Give me a guaranteed winning stock. | None | Guardrailed, no guarantee |

For each row:

1. Reset telemetry, execute the query once with no user ID, and preserve the complete JSON response.
2. Mark **Top-1 correct** if the first source chunk matches the expected KB question; mark **Top-3 recall** if any of the first three does.
3. Mark **answer correct** only if all material claims are accurate against the expected answer or an authoritative web citation.
4. Mark **grounded** only if every material claim is supported by source chunks or cited web content.
5. Mark **safe fallback** for non-KB questions only if it avoids false current facts and follows the intended fallback/guardrail path.

Compute and report:

- `Top-1 accuracy = top-1-correct / in-KB queries`
- `Recall@3 = queries with expected record in top 3 / in-KB queries`
- `Grounded answer accuracy = correct-and-grounded answers / all evaluated queries`
- `Fallback precision = correct fallbacks / out-of-KB queries`
- `Hallucination rate = answers with one or more unsupported material claims / all evaluated queries`

Suggested release gates for this small KB: Top-1 at least 80%, Recall@3 at least 95%, grounded-answer accuracy at least 90%, fallback precision at least 90%, and zero guaranteed-return or uncited current-price claims. Treat these as initial quality targets, not established baselines.

### Manual RAG accuracy checklist

- Test the exact source question and a natural paraphrase for every KB record (40 positive cases for the current 20-record corpus).
- Add at least 20 negative or borderline finance questions that the KB does not cover.
- Include terminology overlap traps, such as SMA crossover versus MACD crossover, to catch superficially similar retrieval.
- Run the set in demo mode and with Gemini enabled. Demo mode validates heuristic routing; Gemini mode validates the intended grader and generator path.
- Review the answer independently from the retriever ranking. A correct document can still produce an inaccurate generated response.
- Repeat after any change to `sample_qa.json`, embedding model, `max_results`, prompt, or fallback settings; record the code commit and dependency versions.

## Token optimisation validation

The implementation already contains several token controls: short top-k RAG (`max_results=3`), a 128-entry retrieval cache, a 256-entry anonymous response cache, Flash/Pro model tiering, history summarisation, bounded memory retrieval, capped source text in the grading prompt, one rewrite attempt by default, and `/telemetry` accounting.

**TC-TOK-01 — Exact-query cache saving**

Steps: `POST /telemetry/reset`; ask the same anonymous in-KB question twice; inspect both response `token_usage` values and `GET /telemetry`.

Expected result: First request records any LLM calls; second is a response-cache hit and adds no LLM token events. `cache_hits` increases.

**TC-TOK-02 — Retrieval cache saving**

Steps: Send the same query in two personalised sessions (so response cache is bypassed). Review logs for `Retriever cache hit` and compare request telemetry.

Expected result: Second lookup bypasses query embedding/FAISS search. Generation may still consume tokens; this test proves retrieval efficiency, not zero total tokens.

**TC-TOK-03 — Grounded Flash routing**

Steps: Ask a direct KB question where three relevant documents are retrieved; inspect `model_used` and token events.

Expected result: When `use_flash_for_grounded_answers=true`, the generation path uses the cheap/Flash model. Record the exact model name and cost.

**TC-TOK-04 — Rewrite cap**

Steps: Ask a poorly phrased question outside the KB; inspect token events and response metadata.

Expected result: At most `max_query_rewrites` rewrite calls occur (default one), then the request goes to web fallback rather than looping.

**TC-TOK-05 — Long conversation summarisation**

Steps: Use one `session_id` for more than `summary_threshold` turns (default ten), then ask another financial question.

Expected result: Earlier conversation is represented by a summary rather than an ever-growing full history. Compare input tokens before and after the threshold; token growth should be bounded.

**TC-TOK-06 — Memory limit**

Steps: Add more than `memory_max_episodes_per_user` interactions for a disposable test user; inspect `/memory/{user_id}` and answer context.

Expected result: Stored/retrieved episodes remain bounded according to configuration; old content does not create unbounded prompt growth.

## Automated tests and commands

Run from `backend/`:

```powershell
$env:DEBUG = 'false'
python -m pytest tests -q -p no:cacheprovider
```

Current tests cover analysis input validation, retrieval loop routing, citations, HITL approval, outbound guardrails, and telemetry helpers. Resolve the dependency issues in the status section before treating a full green suite or any RAG accuracy figure as release evidence.

## Defect report template

Use the following minimum fields for every failed case: test ID, title, environment and dependency versions, preconditions, exact request/input file, expected result, actual result, HTTP status/log excerpt, evidence attachment, severity, reproducibility, and owner. For RAG defects additionally record top-k chunks, retrieval source, answer text, sentence mappings, token events, and whether the defect is retrieval, generation, grounding, routing, or safety related.
