import io
import requests
import pandas as pd
import streamlit as st

API_BASE_URL = " http://127.0.0.1:8000"  # change if needed

st.set_page_config(page_title="Financial Bot", page_icon="📈", layout="wide")

st.title("Financial Bot")
st.caption("Chat & Learn • Buy/Sell Signals")

tab_chat, tab_signals = st.tabs(["💬 Chat / Learn", "📊 Buy/Sell Signals"])

# ---- Chat / Learn ----
with tab_chat:
    st.subheader("Ask a finance question")
    q = st.text_input("Your question", placeholder="e.g., What is P/E ratio?")
    if st.button("Send", type="primary", use_container_width=True):
        if not q:
            st.warning("Please enter a question.")
        else:
            try:
                r = requests.post(f"{API_BASE_URL}/chat", json={"query": q}, timeout=60)
                r.raise_for_status()
                data = r.json()
                st.success(data.get("answer", ""))
                st.caption(f"Confidence: {data.get('confidence', 0):.2f}")
            except Exception as e:
                st.error(f"Chat error: {e}")

# ---- Buy/Sell Signals ----
with tab_signals:
    st.subheader("Upload historical data & get signals")

    col1, col2, col3 = st.columns([1,1,1])
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
                progress.progress(i/len(indicators), text=f"Processing {strategy} ({i}/{len(indicators)})")
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
                        except:
                            pass
                        with results_area:
                            st.error(f"{strategy}: {detail}")
                        continue

                    payload = res.json()
                    buys = payload.get("buy_signals", [])
                    sells = payload.get("sell_signals", [])

                    with results_area:
                        st.markdown(f"### {strategy.replace('_',' ').title()}")
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

st.sidebar.markdown("### Settings")
st.sidebar.code(API_BASE_URL, language="text")
st.sidebar.caption("Change this if your FastAPI runs elsewhere.")

st.markdown("---")
st.caption(
    "Note: PDF parsing uses tabula on the **backend** and requires Java installed where FastAPI runs. "
    "CSV/XLS/XLSX do not require Java."
)
