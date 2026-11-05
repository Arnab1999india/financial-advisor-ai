import io

import pandas as pd
import tabula
from fastapi import UploadFile

from strategies import STRATEGIES


def load_data_from_bytes(ext: str, data: bytes) -> pd.DataFrame:
    """Load OHLC data from validated bytes (CSV, XLS, XLSX, PDF, TXT-as-CSV)."""
    ext = ext.lower()
    buffer = io.BytesIO(data)

    if ext == "csv" or ext == "txt":
        df = pd.read_csv(buffer)
    elif ext in {"xls", "xlsx"}:
        df = pd.read_excel(buffer)
    elif ext == "pdf":
        with open("temp.pdf", "wb") as f:
            f.write(data)
        tables = tabula.read_pdf("temp.pdf", pages="all", multiple_tables=True)
        if not tables:
            raise ValueError("No tables found in the PDF file.")
        df = tables[0]
    else:
        raise ValueError(f"Unsupported file format: {ext}")

    df.rename(columns=lambda c: c.strip().capitalize(), inplace=True)

    required_columns = ["Date", "Open", "High", "Low", "Close"]
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"Input data must contain the following columns: {required_columns}")

    for col in ["Open", "High", "Low", "Close"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df.dropna(subset=required_columns, inplace=True)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df.dropna(subset=["Date"], inplace=True)
    return df


def load_data_from_file(file: UploadFile) -> pd.DataFrame:
    """Legacy loader — prefer validate_upload + load_data_from_bytes in the API layer."""
    file_extension = file.filename.split(".")[-1].lower()
    raw = file.file.read()
    return load_data_from_bytes(file_extension, raw)


def run_analysis(df: pd.DataFrame, strategy_name: str, data_type: str) -> dict:
    """
    Runs the selected trading strategy on the DataFrame.
    """
    if strategy_name not in STRATEGIES:
        raise ValueError(f"Invalid strategy: {strategy_name}")

    strategy_func = STRATEGIES[strategy_name]
    result = strategy_func(df.copy())

    if isinstance(result, pd.DataFrame):
        if "buy_signal" not in result.columns or "sell_signal" not in result.columns:
            raise ValueError("Strategy DataFrame must contain buy_signal and sell_signal columns.")

        buy_signals = result[result["buy_signal"]].to_dict("records")
        sell_signals = result[result["sell_signal"]].to_dict("records")
    elif isinstance(result, dict):
        signal = result.get("signal", "NONE")
        buy_signals = [result] if signal == "BUY" else []
        sell_signals = [result] if signal == "SELL" else []
    else:
        raise ValueError("Strategy must return either a DataFrame or a signal dictionary.")

    return {
        "strategy": strategy_name,
        "data_type": data_type,
        "buy_signals": buy_signals,
        "sell_signals": sell_signals,
    }
