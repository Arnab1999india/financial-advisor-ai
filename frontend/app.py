import html
import io
import uuid
import requests
import pandas as pd
import streamlit as st

API_BASE_URL = "http://127.0.0.1:8000"  # change if needed

st.set_page_config(page_title="Financial Bot", page_icon="📈", layout="wide")

st.title("Financial Bot")
st.caption("Chat & Learn • Buy/Sell Signals • Live Token Telemetry")

tab_chat, tab_signals = st.tabs(["💬 Chat / Learn", "📊 Buy/Sell Signals"])


def _empty_session_telemetry() -> dict:
    return {
        "requests": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "estimated_cost_usd": 0.0,
        "last": None,
    }


if "telemetry_session" not in st.session_state:
    st.session_state.telemetry_session = _empty_session_telemetry()


def _accumulate_session_usage(token_usage: dict | None) -> None:
    if not token_usage:
        return
    session = st.session_state.telemetry_session
    session["requests"] += 1
    session["input_tokens"] += int(token_usage.get("input_tokens") or 0)
    session["output_tokens"] += int(token_usage.get("output_tokens") or 0)
    session["total_tokens"] += int(token_usage.get("total_tokens") or 0)
    session["estimated_cost_usd"] += float(token_usage.get("estimated_cost_usd") or 0.0)
    session["last"] = token_usage


def _fetch_server_telemetry() -> dict | None:
    try:
        res = requests.get(f"{API_BASE_URL}/telemetry", timeout=5)
        res.raise_for_status()
        return res.json()
    except Exception:
        return None


def _usd(value: float) -> str:
    return f"${value:.6f}"


def render_token_usage(token_usage: dict | None, *, heading: str = "This request") -> None:
    if not token_usage:
        st.caption("No token telemetry on this response.")
        return
    st.markdown(f"**{heading}**")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Input tokens", int(token_usage.get("input_tokens") or 0))
    c2.metric("Output tokens", int(token_usage.get("output_tokens") or 0))
    c3.metric("Total tokens", int(token_usage.get("total_tokens") or 0))
    c4.metric("Est. cost", _usd(float(token_usage.get("estimated_cost_usd") or 0.0)))
    events = token_usage.get("events") or []
    if events:
        rows = [
            {
                "Node": e.get("node"),
                "Model": e.get("model"),
                "In": e.get("input_tokens"),
                "Out": e.get("output_tokens"),
                "Total": e.get("total_tokens"),
                "USD": e.get("estimated_cost_usd"),
            }
            for e in events
        ]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def _confidence_color(confidence: float) -> str:
    if confidence >= 0.7:
        return "#bbf7d0"
    if confidence >= 0.45:
        return "#fde68a"
    return "#fecaca"


def highlight_answer_html(answer: str, mappings: list) -> str:
    """Wrap mapped answer spans in <mark> tags tinted by mapping confidence."""
    if not answer:
        return ""
    spans = []
    for item in mappings or []:
        start = int(item.get("answer_start") or 0)
        end = int(item.get("answer_end") or 0)
        if 0 <= start < end <= len(answer):
            spans.append((start, end, float(item.get("confidence") or 0.0)))
    spans.sort(key=lambda s: s[0])

    parts: list[str] = []
    cursor = 0
    for start, end, confidence in spans:
        if start < cursor:
            continue
        parts.append(html.escape(answer[cursor:start]).replace("\n", "<br>"))
        color = _confidence_color(confidence)
        title = f"Grounding confidence {confidence:.0%}"
        inner = html.escape(answer[start:end]).replace("\n", "<br>")
        parts.append(
            f'<mark title="{html.escape(title)}" '
            f'style="background:{color};padding:0.05em 0.15em;border-radius:3px;">'
            f"{inner}</mark>"
        )
        cursor = end
    parts.append(html.escape(answer[cursor:]).replace("\n", "<br>"))
    return "".join(parts)


def render_grounding(data: dict) -> None:
    answer = data.get("answer") or ""
    mappings = data.get("sentence_mappings") or []
    chunks = data.get("source_chunks") or []

    st.markdown("**Answer** (highlighted spans map to retrieved snippets)")
    if mappings:
        st.markdown(highlight_answer_html(answer, mappings), unsafe_allow_html=True)
        st.caption("Highlight colour: green ≥70% · amber ≥45% · red below 45% mapping confidence.")
    else:
        st.success(answer)
        if chunks:
            st.caption("Retrieved snippets are listed below; no sentence mapped above the confidence floor.")

    if mappings:
        map_rows = [
            {
                "Answer sentence": m.get("answer_sentence"),
                "Source sentence": m.get("source_sentence"),
                "Source": m.get("source_title"),
                "Type": m.get("source_type"),
                "Confidence": f"{float(m.get('confidence') or 0):.0%}",
                "Timestamp": m.get("source_timestamp") or "—",
            }
            for m in mappings
        ]
        st.markdown("**Sentence mappings**")
        st.dataframe(pd.DataFrame(map_rows), use_container_width=True, hide_index=True)

    if chunks:
        st.markdown("**Retrieved snippets**")
        for chunk in chunks:
            conf = float(chunk.get("confidence") or 0.0)
            ts = chunk.get("source_timestamp") or chunk.get("retrieved_at") or "unknown"
            retrieved = chunk.get("retrieved_at") or "—"
            title = chunk.get("title") or chunk.get("chunk_id") or "Untitled snippet"
            kind = chunk.get("source_type") or "faiss"
            st.markdown(
                f"**{html.escape(str(title))}** · `{kind}` · "
                f"confidence **{conf:.0%}** · source {html.escape(str(ts))} · retrieved {html.escape(str(retrieved))}"
            )
            snippet = (chunk.get("snippet") or "").strip()
            if snippet:
                st.markdown(
                    f'<div style="background:{_confidence_color(conf)};padding:0.6rem 0.8rem;'
                    f'border-radius:6px;margin-bottom:0.6rem;">{html.escape(snippet)}</div>',
                    unsafe_allow_html=True,
                )
            url = chunk.get("source_url")
            if url:
                st.caption(f"URL: {url}")


def render_chat_result(data: dict) -> None:
    """Render a chat payload, including a paused approval card when present."""
    render_grounding(data)
    confidence = data.get("confidence")
    if confidence is not None:
        st.caption(f"Confidence: {confidence:.2f} · Model: {data.get('model_used') or 'n/a'}")
    else:
        st.caption(f"Model: {data.get('model_used') or 'n/a'}")
    if data.get("guardrail_amended"):
        st.info(
            "Outbound guardrail rewrote this answer into educational phrasing "
            "and appended the statutory risk disclaimer."
        )
    source = data.get("retrieval_source")
    if source == "web":
        st.caption("Context sourced from web search (FAISS docs were not relevant enough).")
    elif source == "faiss_rewritten":
        st.caption("Context sourced from the knowledge base after rewriting the query.")
    elif source == "faiss":
        st.caption("Context sourced from the internal knowledge base.")
    token_usage = data.get("token_usage") or {}
    render_token_usage(token_usage)

    if data.get("status") == "awaiting_approval" and data.get("approval_card"):
        render_approval_card(data)


def render_approval_card(data: dict) -> None:
    card = data.get("approval_card") or {}
    proposal = dict(card.get("proposal") or {})
    reason = card.get("reason") or "pending_approval"
    st.warning(
        "The agent paused this run. Approve, edit, or reject the simulated "
        "trade/alert below — execution does not continue until you decide."
    )
    st.caption(f"Pause reason: `{reason}` · thread `{data.get('thread_id')}`")

    with st.form("hitl_approval_form"):
        st.markdown("**Approval card — edit parameters before confirming**")
        action_type = st.selectbox(
            "Action",
            ["paper_trade", "price_alert"],
            index=0 if proposal.get("action_type") != "price_alert" else 1,
        )
        symbol = st.text_input("Symbol", value=str(proposal.get("symbol") or "DEMO"))
        side = st.selectbox(
            "Side",
            ["BUY", "SELL"],
            index=0 if str(proposal.get("side") or "BUY").upper() != "SELL" else 1,
        )
        quantity = st.number_input(
            "Quantity",
            min_value=0.0,
            value=float(proposal.get("quantity") or 1.0),
            step=1.0,
        )
        order_type = st.selectbox(
            "Order type",
            ["market", "limit"],
            index=0 if str(proposal.get("order_type") or "market") != "limit" else 1,
        )
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            limit_price = st.text_input("Limit price", value=str(proposal.get("limit_price") or ""))
        with col_b:
            stop_loss = st.text_input("Stop loss", value=str(proposal.get("stop_loss") or ""))
        with col_c:
            take_profit = st.text_input("Take profit", value=str(proposal.get("take_profit") or ""))
        alert_trigger = st.text_input(
            "Alert trigger",
            value=str(proposal.get("alert_trigger") or ""),
        )
        notes = st.text_area("Notes (optional)", value="")
        decision = st.radio("Decision", ["Approve", "Reject"], horizontal=True)
        submitted = st.form_submit_button("Submit decision", type="primary")

    if submitted:
        def _num(raw):
            raw = str(raw).strip()
            if not raw:
                return None
            try:
                return float(raw)
            except ValueError:
                return raw

        payload = {
            "thread_id": data.get("thread_id"),
            "decision": "approve" if decision == "Approve" else "reject",
            "notes": notes,
            "params": {
                "action_type": action_type,
                "symbol": symbol.strip().upper() or "DEMO",
                "side": side,
                "quantity": quantity,
                "order_type": order_type,
                "limit_price": _num(limit_price),
                "stop_loss": _num(stop_loss),
                "take_profit": _num(take_profit),
                "alert_trigger": alert_trigger.strip() or None,
                "probability": proposal.get("probability"),
                "rationale": proposal.get("rationale"),
            },
        }
        try:
            res = requests.post(f"{API_BASE_URL}/chat/resume", json=payload, timeout=60)
            res.raise_for_status()
            resumed = res.json()
            st.session_state.last_chat = resumed
            _accumulate_session_usage(resumed.get("token_usage") or {})
            st.rerun()
        except Exception as exc:
            st.error(f"Resume error: {exc}")


# ---- Chat / Learn ----
with tab_chat:
    st.subheader("Ask a finance question")
    q = st.text_input("Your question", placeholder="e.g., What is P/E ratio?")
    if st.button("Send", type="primary", use_container_width=True):
        if not q:
            st.warning("Please enter a question.")
        else:
            try:
                thread_id = str(uuid.uuid4())
                r = requests.post(
                    f"{API_BASE_URL}/chat",
                    json={"query": q, "session_id": thread_id},
                    timeout=60,
                )
                r.raise_for_status()
                data = r.json()
                data["thread_id"] = data.get("thread_id") or thread_id
                st.session_state.last_chat = data
                _accumulate_session_usage(data.get("token_usage") or {})
            except Exception as e:
                st.session_state.last_chat = None
                st.error(f"Chat error: {e}")

    if st.session_state.get("last_chat"):
        render_chat_result(st.session_state.last_chat)

# ---- Buy/Sell Signals ----
with tab_signals:
    st.subheader("Upload historical data & get signals")

    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        data_type = st.selectbox("Data Type", ["swing", "intraday"])
    with col2:
        primary = st.selectbox(
            "Primary Indicator",
            [
                "swing_trading",
                "scalping_trading",
                "44_moving_average",
                "trend_trading",
                "range_trading",
            ],
            index=0,
        )
    with col3:
        extras = st.multiselect(
            "Additional Indicators (optional)",
            [
                "swing_trading",
                "scalping_trading",
                "44_moving_average",
                "trend_trading",
                "range_trading",
            ],
        )

    file = st.file_uploader(
        "Historical data file (CSV / XLS / XLSX / PDF)",
        type=["csv", "xls", "xlsx", "pdf"],
        accept_multiple_files=False,
        help="Required columns: Date, Open, High, Low, Close. Max ~10MB (your backend can enforce).",
    )

    run = st.button("Analyze & Get Signals", type="primary")
    if run:
        if not file:
            st.warning("Please upload a file.")
        else:
            # Build indicator list (unique)
            indicators = list(dict.fromkeys([primary] + extras))
            progress = st.progress(0.0, text="Starting…")
            results_area = st.container()

            # Prepare upload for requests
            file_bytes = file.read()
            filename = file.name
            for i, strategy in enumerate(indicators, start=1):
                progress.progress(i / len(indicators), text=f"Processing {strategy} ({i}/{len(indicators)})")
                files = {"file": (filename, io.BytesIO(file_bytes), file.type or "application/octet-stream")}
                data = {"strategy": strategy, "data_type": data_type}

                try:
                    res = requests.post(f"{API_BASE_URL}/analyze", files=files, data=data, timeout=120)
                    if res.status_code != 200:
                        # try to parse backend error
                        detail = f"HTTP {res.status_code}"
                        try:
                            j = res.json()
                            if "detail" in j:
                                detail = j["detail"]
                        except Exception:
                            pass
                        with results_area:
                            st.error(f"{strategy}: {detail}")
                        continue

                    payload = res.json()
                    buys = payload.get("buy_signals", [])
                    sells = payload.get("sell_signals", [])

                    with results_area:
                        st.markdown(f"### {strategy.replace('_', ' ').title()}")
                        kpi1, kpi2 = st.columns(2)
                        kpi1.metric("Buy Signals", len(buys))
                        kpi2.metric("Sell Signals", len(sells))

                        # Tables
                        if buys:
                            st.markdown("**Buy Signals**")
                            st.dataframe(pd.DataFrame(buys), use_container_width=True)
                        else:
                            st.info("No buy signals.")

                        if sells:
                            st.markdown("**Sell Signals**")
                            st.dataframe(pd.DataFrame(sells), use_container_width=True)
                        else:
                            st.info("No sell signals.")
                except Exception as e:
                    with results_area:
                        st.error(f"{strategy}: {e}")

            progress.empty()
            st.success("Done.")

# ---- Live Token Telemetry (sidebar) ----
st.sidebar.markdown("### Settings")
st.sidebar.code(API_BASE_URL, language="text")
st.sidebar.caption("Change this if your FastAPI runs elsewhere.")

st.sidebar.divider()
st.sidebar.markdown("### Live Token Telemetry")

session = st.session_state.telemetry_session
st.sidebar.caption("This Streamlit session")
s1, s2 = st.sidebar.columns(2)
s1.metric("Requests", session["requests"])
s2.metric("Tokens", session["total_tokens"])
s3, s4 = st.sidebar.columns(2)
s3.metric("Input", session["input_tokens"])
s4.metric("Output", session["output_tokens"])
st.sidebar.metric("Session cost", _usd(session["estimated_cost_usd"]))

if session["last"]:
    with st.sidebar.expander("Last request breakdown", expanded=True):
        last = session["last"]
        st.write(
            f"{last.get('input_tokens', 0)} in / {last.get('output_tokens', 0)} out "
            f"({last.get('calls', 0)} LLM call(s), {_usd(float(last.get('estimated_cost_usd') or 0))})"
        )
        by_node = last.get("by_node") or {}
        if by_node:
            st.caption("By node")
            for node, bucket in by_node.items():
                st.write(f"- `{node}`: {bucket.get('total_tokens', 0)} tokens")

st.sidebar.caption("FastAPI process (all clients)")
server = _fetch_server_telemetry()
if server is None:
    st.sidebar.warning("Backend telemetry unreachable.")
else:
    st.sidebar.metric("Server tokens", server.get("total_tokens", 0))
    st.sidebar.metric("Server cost", _usd(float(server.get("estimated_cost_usd") or 0)))
    st.sidebar.caption(
        f"{server.get('chat_requests', 0)} chats · "
        f"{server.get('llm_calls', 0)} LLM calls · "
        f"{server.get('cache_hits', 0)} cache hits"
    )
    by_model = server.get("by_model") or {}
    if by_model:
        with st.sidebar.expander("By model"):
            for model, bucket in by_model.items():
                st.write(
                    f"**{model}** — {bucket.get('calls', 0)} calls, "
                    f"{bucket.get('total_tokens', 0)} tok, "
                    f"{_usd(float(bucket.get('estimated_cost_usd') or 0))}"
                )
    recent = server.get("recent_calls") or []
    if recent:
        with st.sidebar.expander("Recent LLM calls"):
            preview = [
                {
                    "node": e.get("node"),
                    "model": e.get("model"),
                    "tokens": e.get("total_tokens"),
                }
                for e in recent[:10]
            ]
            st.dataframe(pd.DataFrame(preview), use_container_width=True, hide_index=True)

reset_cols = st.sidebar.columns(2)
if reset_cols[0].button("Reset session"):
    st.session_state.telemetry_session = _empty_session_telemetry()
    st.rerun()
if reset_cols[1].button("Reset server"):
    try:
        requests.post(f"{API_BASE_URL}/telemetry/reset", timeout=5).raise_for_status()
        st.rerun()
    except Exception as e:
        st.sidebar.error(f"Reset failed: {e}")

st.markdown("---")
st.caption(
    "Note: PDF parsing uses tabula on the **backend** and requires Java installed where FastAPI runs. "
    "CSV/XLS/XLSX do not require Java. Token telemetry is estimated from Gemini usage metadata."
)
