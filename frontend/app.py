import html
import inspect
import io
import json
import sys
import uuid
from pathlib import Path

import pandas as pd
import requests
import streamlit as st
import streamlit.components.v1 as components

API_BASE_URL = "http://127.0.0.1:8000"  # change if needed
_FRONTEND_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _FRONTEND_DIR.parent
_BACKEND_DIR = _REPO_ROOT / "backend"
if _BACKEND_DIR.is_dir() and str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

try:
    from tools.market_data import SUPPORTED_EQUITY_EXCHANGES
except ImportError:  # pragma: no cover - fallback if backend path unavailable
    SUPPORTED_EQUITY_EXCHANGES = (
        {
            "id": "NSE",
            "label": "NSE — National Stock Exchange of India",
            "yfinance_suffix": ".NS",
        },
        {
            "id": "BSE",
            "label": "BSE — Bombay Stock Exchange",
            "yfinance_suffix": ".BO",
        },
    )

_SIGNAL_TRADING_STYLE_SWING = "Swing Trading"
_SIGNAL_TRADING_STYLE_INTRADAY = "Intraday"
_SIGNAL_TRADING_STYLE_OPTIONS = (
    _SIGNAL_TRADING_STYLE_SWING,
    _SIGNAL_TRADING_STYLE_INTRADAY,
)
_SIGNAL_STYLE_TO_DATA_TYPE = {
    _SIGNAL_TRADING_STYLE_SWING: "swing",
    _SIGNAL_TRADING_STYLE_INTRADAY: "intraday",
}
_SIGNAL_SWING_INTERVALS = ("1 day (Daily)",)
_SIGNAL_INTRADAY_INTERVALS = ("1 minute", "5 minutes")
_SIGNAL_DEFAULT_INTERVAL = {
    _SIGNAL_TRADING_STYLE_SWING: "1 day (Daily)",
    _SIGNAL_TRADING_STYLE_INTRADAY: "5 minutes",
}
_SIGNAL_STRATEGY_OPTIONS = [
    "swing_trading",
    "scalping_trading",
    "44_moving_average",
    "trend_trading",
    "range_trading",
]


def _signal_interval_options(trading_style: str) -> tuple[str, ...]:
    if trading_style == _SIGNAL_TRADING_STYLE_INTRADAY:
        return _SIGNAL_INTRADAY_INTERVALS
    return _SIGNAL_SWING_INTERVALS


def _on_signals_trading_style_change() -> None:
    style = st.session_state.signals_trading_style
    st.session_state.signals_data_interval = _SIGNAL_DEFAULT_INTERVAL[style]


def _init_signals_session_state() -> None:
    if "signals_trading_style" not in st.session_state:
        st.session_state.signals_trading_style = _SIGNAL_TRADING_STYLE_SWING
    if "signals_data_interval" not in st.session_state:
        st.session_state.signals_data_interval = _SIGNAL_DEFAULT_INTERVAL[
            st.session_state.signals_trading_style
        ]
    if "signals_exchange_id" not in st.session_state:
        st.session_state.signals_exchange_id = SUPPORTED_EQUITY_EXCHANGES[0]["id"]
    if "signals_instrument" not in st.session_state:
        st.session_state.signals_instrument = ""
    if "signals_primary" not in st.session_state:
        st.session_state.signals_primary = _SIGNAL_STRATEGY_OPTIONS[0]
    if "signals_extras" not in st.session_state:
        st.session_state.signals_extras = []
    if "signals_upload_bytes" not in st.session_state:
        st.session_state.signals_upload_bytes = None
    if "signals_upload_name" not in st.session_state:
        st.session_state.signals_upload_name = ""
    if "signals_upload_type" not in st.session_state:
        st.session_state.signals_upload_type = "application/octet-stream"
    if "signals_analysis_busy" not in st.session_state:
        st.session_state.signals_analysis_busy = False
    if "signals_last_results" not in st.session_state:
        st.session_state.signals_last_results = []


def _exchange_by_id(exchange_id: str) -> dict[str, str]:
    for row in SUPPORTED_EQUITY_EXCHANGES:
        if row["id"] == exchange_id:
            return row
    return SUPPORTED_EQUITY_EXCHANGES[0]


def _signals_market_ticker(instrument: str, exchange_id: str) -> str:
    symbol = (instrument or "").strip().upper()
    if not symbol:
        return ""
    suffix = _exchange_by_id(exchange_id)["yfinance_suffix"]
    return f"{symbol}{suffix}"


def _validate_signals_inputs() -> list[str]:
    errors: list[str] = []
    style = st.session_state.get("signals_trading_style")
    if style not in _SIGNAL_TRADING_STYLE_OPTIONS:
        errors.append("Select a **Trading Style** (Swing Trading or Intraday).")

    instrument = (st.session_state.get("signals_instrument") or "").strip()
    if not instrument:
        errors.append("Enter an **Instrument or Symbol** (e.g. TCS).")

    exchange_id = st.session_state.get("signals_exchange_id")
    if not exchange_id or exchange_id not in {row["id"] for row in SUPPORTED_EQUITY_EXCHANGES}:
        errors.append("Select an **Exchange** (NSE or BSE).")

    interval = (st.session_state.get("signals_data_interval") or "").strip()
    if not interval:
        errors.append("Choose a **Data Interval**.")

    if not st.session_state.get("signals_upload_bytes"):
        errors.append("Upload a **historical data file** (CSV, XLS, XLSX, or PDF).")

    primary = st.session_state.get("signals_primary")
    if not primary or primary not in _SIGNAL_STRATEGY_OPTIONS:
        errors.append("Select a **Primary Indicator**.")

    return errors


def _signals_indicator_list() -> list[str]:
    primary = st.session_state.signals_primary
    extras = st.session_state.signals_extras or []
    return list(dict.fromkeys([primary] + list(extras)))


def _signals_analysis_form_data() -> dict[str, str]:
    style = st.session_state.signals_trading_style
    return {
        "data_type": _SIGNAL_STYLE_TO_DATA_TYPE[style],
        "symbol": (st.session_state.signals_instrument or "").strip().upper(),
        "exchange": st.session_state.signals_exchange_id,
        "data_interval": (st.session_state.signals_data_interval or "").strip(),
    }


def _render_signals_payload(payload: dict, *, container) -> None:
    strategy = payload.get("strategy", "strategy")
    buys = payload.get("buy_signals", [])
    sells = payload.get("sell_signals", [])

    with container:
        st.markdown(f"### {strategy.replace('_', ' ').title()}")
        meta_bits = []
        if payload.get("symbol"):
            meta_bits.append(f"**Symbol:** {payload['symbol']}")
        if payload.get("exchange"):
            meta_bits.append(f"**Exchange:** {payload['exchange']}")
        if payload.get("data_interval"):
            meta_bits.append(f"**Interval:** {payload['data_interval']}")
        if payload.get("data_type"):
            meta_bits.append(f"**Data type:** `{payload['data_type']}`")
        if meta_bits:
            st.caption(" · ".join(meta_bits))

        kpi1, kpi2 = st.columns(2)
        kpi1.metric("Buy Signals", len(buys))
        kpi2.metric("Sell Signals", len(sells))

        if buys:
            st.markdown("**Buy Signals**")
            buy_df = pd.DataFrame(buys)
            st.dataframe(buy_df, use_container_width=True)
            if {"Date", "Close"}.issubset(buy_df.columns):
                st.line_chart(buy_df.set_index("Date")["Close"])
        else:
            st.info("No buy signals.")

        if sells:
            st.markdown("**Sell Signals**")
            sell_df = pd.DataFrame(sells)
            st.dataframe(sell_df, use_container_width=True)
            if {"Date", "Close"}.issubset(sell_df.columns):
                st.line_chart(sell_df.set_index("Date")["Close"])
        else:
            st.info("No sell signals.")


def _render_signals_results(results: list[dict]) -> None:
    if not results:
        return
    st.divider()
    st.markdown("#### Results")
    area = st.container()
    for payload in results:
        _render_signals_payload(payload, container=area)


def _run_signals_analysis() -> None:
    indicators = _signals_indicator_list()
    form_meta = _signals_analysis_form_data()
    file_bytes = st.session_state.signals_upload_bytes
    filename = st.session_state.signals_upload_name
    file_type = st.session_state.signals_upload_type

    progress = st.progress(0.0, text="Starting…")
    st.session_state.signals_last_results = []
    had_error = False

    files = {"file": (filename, io.BytesIO(file_bytes), file_type)}
    data = {"strategies": ",".join(indicators), **form_meta}

    try:
        progress.progress(0.5, text=f"Processing {len(indicators)} indicator(s)…")
        res = requests.post(f"{API_BASE_URL}/analyze/batch", files=files, data=data, timeout=300)
        if res.status_code != 200:
            had_error = True
            detail = f"HTTP {res.status_code}"
            try:
                err_payload = res.json()
                if "detail" in err_payload:
                    detail = err_payload["detail"]
            except Exception:
                pass
            st.error(detail)
        else:
            st.session_state.signals_last_results = res.json()
    except Exception as exc:
        had_error = True
        st.error(str(exc))

    progress.empty()
    if st.session_state.signals_last_results:
        st.success("Done.")
    elif not had_error:
        st.warning("No results returned.")

_CHAT_INPUT = components.declare_component(
    "chat_input",
    path=str(_FRONTEND_DIR / "chat_input_component"),
)

# Voice/mic Streamlit component — paused (re-enable when mic UX is fixed).
# _VOICE_CHAT_INPUT = components.declare_component(
#     "voice_chat_input",
#     path=str(_FRONTEND_DIR / "voice_chat_component"),
# )

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


def _escape_script(js: str) -> str:
    return js.replace("</script>", "<\\/script>")


def _render_chat_input(*, busy: bool) -> dict | str | None:
    """Textarea with Enter to send, Shift+Enter for newline, and Send button."""
    return _CHAT_INPUT(
        busy=busy,
        key="chat_input_widget",
    )


# def _render_streamlit_voice_input(*, disabled: bool, clear_input: bool) -> dict | str | None:
#     """Bidirectional voice chat input (mic + Enter/Shift+Enter + Send)."""
#     return _VOICE_CHAT_INPUT(
#         disabled=disabled,
#         clear_input=clear_input,
#         key="voice_chat_input_widget",
#     )


def _iter_chat_stream(query: str, thread_id: str):
    """Yield answer token chunks from ``POST /chat/stream`` (SSE)."""
    with requests.post(
        f"{API_BASE_URL}/chat/stream",
        json={"query": query, "session_id": thread_id},
        stream=True,
        timeout=120,
        headers={"Accept": "text/event-stream"},
    ) as response:
        response.raise_for_status()
        buffer = ""
        for chunk in response.iter_content(chunk_size=None, decode_unicode=True):
            if not chunk:
                continue
            buffer += chunk
            while "\n\n" in buffer:
                block, buffer = buffer.split("\n\n", 1)
                for line in block.splitlines():
                    if not line.startswith("data: "):
                        continue
                    payload = json.loads(line[6:])
                    event_type = payload.get("type")
                    if event_type == "status":
                        yield ("__status__", payload.get("message") or "")
                    elif event_type == "token":
                        yield ("__token__", payload.get("text") or "")
                    elif event_type == "done":
                        yield ("__done__", payload.get("response") or {})
                    elif event_type == "error":
                        raise RuntimeError(payload.get("message") or "Stream error")


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

    if "chat_busy" not in st.session_state:
        st.session_state.chat_busy = False
    chat_event = _render_chat_input(
        busy=st.session_state.chat_busy,
    )

    q: str | None = None
    if isinstance(chat_event, dict) and chat_event.get("action") == "send":
        q = (chat_event.get("text") or "").strip()

    if q is not None:
        if not q:
            st.warning("Please enter a question.")
        elif st.session_state.chat_busy:
            st.info("Please wait for the current answer to finish.")
        else:
            st.session_state.last_chat = None
            st.session_state.chat_busy = True
            try:
                thread_id = str(uuid.uuid4())
                status = st.status("Connecting…", expanded=True)
                answer_parts: list[str] = []
                answer_placeholder = st.empty()

                for kind, value in _iter_chat_stream(q, thread_id):
                    if kind == "__status__":
                        status.update(label=value)
                    elif kind == "__token__":
                        answer_parts.append(value)
                        answer_placeholder.markdown("".join(answer_parts))
                    elif kind == "__done__":
                        data = dict(value)
                        data["thread_id"] = data.get("thread_id") or thread_id
                        st.session_state.last_chat = data
                        _accumulate_session_usage(data.get("token_usage") or {})
                        status.update(label="Done", state="complete")
                        answer_placeholder.empty()
            except Exception as e:
                st.session_state.last_chat = None
                st.error(f"Chat error: {e}")
            finally:
                st.session_state.chat_busy = False

    if st.session_state.get("last_chat"):
        render_chat_result(st.session_state.last_chat)

# ---- Buy/Sell Signals ----
with tab_signals:
    st.subheader("Upload historical data & get signals")
    _init_signals_session_state()

    st.markdown("### Configure Your Analysis")

    cfg1, cfg2, cfg3 = st.columns([1, 1, 1])
    with cfg1:
        trading_style = st.selectbox(
            "Trading Style *",
            _SIGNAL_TRADING_STYLE_OPTIONS,
            key="signals_trading_style",
            on_change=_on_signals_trading_style_change,
            help="Required. Matches the `data_type` sent with your upload (`swing` or `intraday`).",
        )
    with cfg2:
        st.text_input(
            "Instrument or Symbol",
            key="signals_instrument",
            placeholder="e.g. TCS, RELIANCE",
            help="Ticker symbol without exchange suffix (same convention as live market quotes).",
        )
    with cfg3:
        exchange_ids = [row["id"] for row in SUPPORTED_EQUITY_EXCHANGES]
        st.selectbox(
            "Exchange",
            exchange_ids,
            format_func=lambda eid: _exchange_by_id(eid)["label"],
            key="signals_exchange_id",
            help="Venues supported by the project's market-data resolver (yfinance suffixes).",
        )

    instrument_raw = (st.session_state.signals_instrument or "").strip().upper()
    exchange_row = _exchange_by_id(st.session_state.signals_exchange_id)
    if instrument_raw:
        st.caption(
            f"Market-data ticker preview: `{instrument_raw}{exchange_row['yfinance_suffix']}` "
            f"({exchange_row['id']})"
        )

    interval_options = list(_signal_interval_options(trading_style))

    interval_kwargs: dict = {
        "label": "Data Interval",
        "options": interval_options,
        "key": "signals_data_interval",
        "help": "Suggested default updates when Trading Style changes; you can pick another allowed interval.",
    }
    if "accept_new_options" in inspect.signature(st.selectbox).parameters:
        interval_kwargs["accept_new_options"] = True
    st.selectbox(**interval_kwargs)

    if trading_style == _SIGNAL_TRADING_STYLE_SWING:
        st.info(
            "Daily OHLCV candles are expected for best results. Swing setups typically use "
            "daily trends and hold positions for multiple trading sessions."
        )
    else:
        st.info(
            "1-minute or 5-minute OHLCV candles are expected for best results. Intraday strategies "
            "analyze price movements within a trading session."
        )

    data_type = _SIGNAL_STYLE_TO_DATA_TYPE[trading_style]

    st.divider()
    st.markdown("**Indicators**")

    col1, col2 = st.columns([1, 1])
    with col1:
        st.selectbox(
            "Primary Indicator",
            _SIGNAL_STRATEGY_OPTIONS,
            key="signals_primary",
        )
    with col2:
        st.multiselect(
            "Additional Indicators (optional)",
            _SIGNAL_STRATEGY_OPTIONS,
            key="signals_extras",
        )

    uploaded_file = st.file_uploader(
        "Historical data file (CSV / XLS / XLSX / PDF)",
        type=["csv", "xls", "xlsx", "pdf"],
        accept_multiple_files=False,
        key="signals_upload_widget",
        help="Required columns: Date, Open, High, Low, Close. Max ~10MB (your backend can enforce).",
    )
    if uploaded_file is not None:
        st.session_state.signals_upload_bytes = uploaded_file.getvalue()
        st.session_state.signals_upload_name = uploaded_file.name
        st.session_state.signals_upload_type = uploaded_file.type or "application/octet-stream"
    else:
        st.session_state.signals_upload_bytes = None
        st.session_state.signals_upload_name = ""
        st.session_state.signals_upload_type = "application/octet-stream"

    if st.session_state.signals_upload_name:
        st.caption(f"Selected file: **{st.session_state.signals_upload_name}**")

    ticker = _signals_market_ticker(
        st.session_state.signals_instrument,
        st.session_state.signals_exchange_id,
    )
    st.markdown("#### Analysis Summary")
    summary_rows = {
        "Trading style": trading_style,
        "Data type (API)": data_type,
        "Instrument": (st.session_state.signals_instrument or "").strip().upper() or "—",
        "Exchange": st.session_state.signals_exchange_id,
        "Ticker": ticker or "—",
        "Data interval": st.session_state.signals_data_interval,
        "Primary indicator": st.session_state.signals_primary.replace("_", " ").title(),
        "Additional indicators": ", ".join(
            s.replace("_", " ").title() for s in (st.session_state.signals_extras or [])
        )
        or "None",
        "Upload": st.session_state.signals_upload_name or "—",
    }
    st.table(pd.DataFrame({"Setting": list(summary_rows.keys()), "Value": list(summary_rows.values())}))

    run = st.button(
        "Analyze & Get Signals",
        type="primary",
        disabled=st.session_state.signals_analysis_busy,
    )
    if st.session_state.signals_analysis_busy:
        st.caption("Analysis in progress…")

    if run:
        if st.session_state.signals_analysis_busy:
            st.info("Please wait for the current analysis to finish.")
        else:
            validation_errors = _validate_signals_inputs()
            if validation_errors:
                for msg in validation_errors:
                    st.warning(msg)
            else:
                st.session_state.signals_analysis_busy = True
                try:
                    _run_signals_analysis()
                finally:
                    st.session_state.signals_analysis_busy = False

    if st.session_state.signals_last_results and not st.session_state.signals_analysis_busy:
        _render_signals_results(st.session_state.signals_last_results)

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
