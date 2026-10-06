import pandas as pd
from fastapi import UploadFile
import tabula
import io

from strategies import STRATEGIES

def load_data_from_file(file: UploadFile) -> pd.DataFrame:
    """
    Loads data from an uploaded file (CSV, XLS, or PDF) into a pandas DataFrame,
    and performs basic data cleaning.
    """
    file_extension = file.filename.split('.')[-1].lower()

    if file_extension == 'csv':
        df = pd.read_csv(file.file)
    elif file_extension in ['xls', 'xlsx']:
        df = pd.read_excel(file.file)
    elif file_extension == 'pdf':
        # For PDF, we need to save it temporarily to use tabula
        with open("temp.pdf", "wb") as f:
            f.write(file.file.read())
        tables = tabula.read_pdf("temp.pdf", pages='all', multiple_tables=True)
        if not tables:
            raise ValueError("No tables found in the PDF file.")
        df = tables[0]
    else:
        raise ValueError(f"Unsupported file format: {file_extension}")

    # Standardize column names to be case-insensitive and stripped
    df.rename(columns=lambda c: c.strip().capitalize(), inplace=True)

    # --- Data Cleaning ---
    required_columns = ['Date', 'Open', 'High', 'Low', 'Close']
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"Input data must contain the following columns: {required_columns}")
    
    # Ensure numeric types for OHLC columns, coercing errors will result in NaN
    for col in ['Open', 'High', 'Low', 'Close']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Drop rows with missing values in key columns
    df.dropna(subset=required_columns, inplace=True)
    
    # Ensure 'Date' column is in datetime format
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    
    # Drop rows where date conversion failed
    df.dropna(subset=['Date'], inplace=True)

    return df


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
        "sell_signals": sell_signals
    }
