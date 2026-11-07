"""OHLCV ingestion helpers and data-quality pipeline.

The pipeline preserves the uploaded source columns while standardizing known
NSE/Yahoo-style headers into a consistent internal naming convention.
"""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Any, BinaryIO

import pandas as pd
from fastapi import UploadFile


# Canonical names used by the analysis engine.
_REQUIRED_COLUMNS = ("Date", "Open", "High", "Low", "Close", "Volume")

# Known source-header variations. Unknown columns are preserved with a
# normalized, readable name rather than being discarded.
_COLUMN_ALIASES = {
    "date": "Date",
    "datetime": "Date",
    "timestamp": "Date",
    "series": "Series",
    "open": "Open",
    "high": "High",
    "low": "Low",
    "close": "Close",
    "ltp": "LTP",
    "prev close": "Prev Close",
    "previous close": "Prev Close",
    "vwap": "VWAP",
    "52 week high": "52 Week High",
    "52 week low": "52 Week Low",
    "volume": "Volume",
    "value": "Value",
    "no of trades": "Number Of Trades",
    "number of trades": "Number Of Trades",
}

_NUMERIC_COLUMNS = {
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
    "LTP",
    "Prev Close",
    "VWAP",
    "52 Week High",
    "52 Week Low",
    "Value",
    "Number Of Trades",
}


def _header_key(column: Any) -> str:
    """Convert a source header to a comparable key."""
    text = " ".join(str(column).strip().replace(".", " ").split())
    return text.casefold()


def _normalize_extension(ext: str | None, filename: str | None = None) -> str:
    """Resolve and validate a supported file extension."""
    candidate = ext or (Path(filename).suffix if filename else "")
    candidate = candidate.lower().strip()
    if candidate and not candidate.startswith("."):
        candidate = f".{candidate}"

    if candidate not in {".csv", ".xlsx", ".xls"}:
        raise ValueError(
            "Unsupported file type. Please upload a CSV or Excel file (.csv, .xlsx, .xls)."
        )
    return candidate


def _read_source_bytes(
    file: UploadFile | None,
    path: str | Path | None,
    data: bytes | None,
) -> tuple[bytes, str | None]:
    """Read exactly one input source and return its bytes and filename."""
    supplied = sum(source is not None for source in (file, path, data))
    if supplied != 1:
        raise ValueError("Provide exactly one input source: file, path, or data.")

    if data is not None:
        if not isinstance(data, bytes) or not data:
            raise ValueError("The supplied file data must be non-empty bytes.")
        return data, None

    if path is not None:
        source_path = Path(path)
        if not source_path.exists() or not source_path.is_file():
            raise FileNotFoundError(f"OHLCV file not found: {source_path}")
        return source_path.read_bytes(), source_path.name

    assert file is not None
    stream: BinaryIO = file.file
    try:
        current_position = stream.tell()
    except (AttributeError, OSError):
        current_position = None

    try:
        stream.seek(0)
        content = stream.read()
    finally:
        if current_position is not None:
            try:
                stream.seek(current_position)
            except (AttributeError, OSError):
                pass

    if not content:
        raise ValueError("The uploaded file is empty.")
    return content, file.filename


def load_ohlcv_file(
    file: UploadFile | None = None,
    *,
    path: str | Path | None = None,
    ext: str | None = None,
    data: bytes | None = None,
) -> pd.DataFrame:
    """Load raw OHLCV rows from an upload, filesystem path, or validated bytes.

    Supports CSV and Excel files. The source is loaded without changing its
    column names or sorting its rows.
    """
    content, filename = _read_source_bytes(file, path, data)
    extension = _normalize_extension(ext, filename)

    try:
        buffer = BytesIO(content)
        if extension == ".csv":
            return pd.read_csv(buffer)
        return pd.read_excel(buffer)
    except ImportError as exc:
        raise RuntimeError(
            "Excel support requires an installed pandas Excel engine, such as openpyxl."
        ) from exc
    except Exception as exc:
        raise ValueError(f"Could not read the {extension} file: {exc}") from exc


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize known source headers while preserving all columns and rows."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")

    result = df.copy()
    normalized_names: list[str] = []
    for original in result.columns:
        key = _header_key(original)
        if key in _COLUMN_ALIASES:
            normalized_names.append(_COLUMN_ALIASES[key])
        else:
            readable = " ".join(str(original).strip().split())
            normalized_names.append(readable)

    if len(set(normalized_names)) != len(normalized_names):
        duplicates = sorted(
            {
                name
                for name in normalized_names
                if normalized_names.count(name) > 1
            }
        )
        raise ValueError(
            "Column normalization produced duplicate column names: "
            + ", ".join(duplicates)
        )

    result.columns = normalized_names
    return result


def convert_data_types(df: pd.DataFrame) -> pd.DataFrame:
    """Convert Date and recognized numeric columns, rejecting bad non-empty values.

    Commas are removed from numeric strings to support Indian-formatted values
    such as ``6,54,01,905``. Missing values remain missing for later reporting.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")

    result = df.copy()

    if "Date" in result.columns:
        original = result["Date"]
        parsed = pd.to_datetime(original, errors="coerce", dayfirst=True)
        invalid = original.notna() & original.astype(str).str.strip().ne("") & parsed.isna()
        if invalid.any():
            examples = original[invalid].astype(str).head(5).tolist()
            raise ValueError(f"Invalid date values in 'Date': {examples}")
        result["Date"] = parsed

    for column in _NUMERIC_COLUMNS.intersection(result.columns):
        original = result[column]
        text = original.astype("string").str.strip()
        text = text.str.replace(",", "", regex=False)
        text = text.replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})
        converted = pd.to_numeric(text, errors="coerce")

        invalid = original.notna() & text.notna() & converted.isna()
        if invalid.any():
            examples = original[invalid].astype(str).head(5).tolist()
            raise ValueError(
                f"Non-numeric values found in '{column}': {examples}"
            )

        result[column] = converted

    return result


def validate_required_columns(
    df: pd.DataFrame,
    required: tuple[str, ...] = _REQUIRED_COLUMNS,
) -> list[str]:
    """Return names of required columns that are missing from the frame."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")
    return [column for column in required if column not in df.columns]


def validate_ohlcv(df: pd.DataFrame) -> None:
    """Raise ValueError when required OHLCV rows violate structural constraints.

    Missing values and invalid rows are reported with their DataFrame index.
    The function does not modify the input DataFrame.
    """
    missing_columns = validate_required_columns(df)
    if missing_columns:
        raise ValueError(
            "Missing required OHLCV columns: " + ", ".join(missing_columns)
        )

    issues: list[str] = []
    required = list(_REQUIRED_COLUMNS)

    null_mask = df[required].isna().any(axis=1)
    if null_mask.any():
        issues.append(
            f"{int(null_mask.sum())} row(s) contain missing required OHLCV values "
            f"(indexes: {df.index[null_mask].tolist()[:10]})."
        )

    numeric_columns = ["Open", "High", "Low", "Close", "Volume"]
    for column in numeric_columns:
        non_numeric = ~pd.to_numeric(df[column], errors="coerce").notna()
        # Missing values are already reported above; this catches non-numeric data.
        if (non_numeric & df[column].notna()).any():
            issues.append(f"Column '{column}' contains non-numeric values.")

    price_columns = ["Open", "High", "Low", "Close"]
    invalid_price = (df[price_columns] <= 0).any(axis=1)
    if invalid_price.any():
        issues.append(
            f"{int(invalid_price.sum())} row(s) contain zero or negative OHLC prices."
        )

    invalid_volume = df["Volume"] < 0
    if invalid_volume.any():
        issues.append(
            f"{int(invalid_volume.sum())} row(s) contain negative volume."
        )

    invalid_high = (
        (df["High"] < df[["Open", "Close", "Low"]].max(axis=1))
    )
    invalid_low = (
        (df["Low"] > df[["Open", "Close", "High"]].min(axis=1))
    )
    invalid_ohlc = invalid_high | invalid_low
    if invalid_ohlc.any():
        issues.append(
            f"{int(invalid_ohlc.sum())} row(s) violate OHLC high/low relationships "
            f"(indexes: {df.index[invalid_ohlc].tolist()[:10]})."
        )

    if "Date" in df.columns:
        invalid_dates = df["Date"].isna()
        if invalid_dates.any():
            issues.append(f"{int(invalid_dates.sum())} row(s) have invalid or missing dates.")

    if issues:
        raise ValueError("OHLCV validation failed: " + " ".join(issues))


def sort_candles(
    df: pd.DataFrame,
    *,
    date_column: str = "Date",
) -> pd.DataFrame:
    """Sort candles by date in ascending chronological order."""
    if date_column not in df.columns:
        raise ValueError(f"Date column '{date_column}' is missing.")
    result = df.copy()
    result = result.sort_values(date_column, ascending=True, kind="stable")
    return result.reset_index(drop=True)


def remove_duplicate_candles(
    df: pd.DataFrame,
    *,
    date_column: str = "Date",
) -> pd.DataFrame:
    """Drop duplicate daily candle dates, keeping the last occurrence.

    This policy is appropriate only when one row per symbol/date is expected.
    Duplicate counts are captured by the quality report before this function
    is called in the main pipeline.
    """
    if date_column not in df.columns:
        raise ValueError(f"Date column '{date_column}' is missing.")
    result = df.copy()
    return (
        result.drop_duplicates(subset=[date_column], keep="last")
        .reset_index(drop=True)
    )


def check_missing_values(df: pd.DataFrame) -> dict[str, int]:
    """Count null values per column relevant to OHLCV analysis."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")
    relevant = [column for column in df.columns if column in _NUMERIC_COLUMNS or column == "Date"]
    return {column: int(df[column].isna().sum()) for column in relevant}


def generate_quality_report(df: pd.DataFrame) -> dict[str, Any]:
    """Build a structured summary of completeness and candle-data quality."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")

    missing = check_missing_values(df)
    duplicate_count = (
        int(df["Date"].duplicated(keep=False).sum()) if "Date" in df.columns else 0
    )

    invalid_ohlc_count = 0
    if all(column in df.columns for column in ("Open", "High", "Low", "Close")):
        prices = df[["Open", "High", "Low", "Close"]]
        invalid_price = (prices <= 0).any(axis=1)
        invalid_high = df["High"] < prices[["Open", "Close", "Low"]].max(axis=1)
        invalid_low = df["Low"] > prices[["Open", "Close", "High"]].min(axis=1)
        invalid_ohlc_count = int((invalid_price | invalid_high | invalid_low).sum())

    date_min = date_max = None
    if "Date" in df.columns and df["Date"].notna().any():
        date_min = pd.Timestamp(df["Date"].min()).isoformat()
        date_max = pd.Timestamp(df["Date"].max()).isoformat()

    missing_required = validate_required_columns(df)
    required_missing_rows = (
        int(df[list(_REQUIRED_COLUMNS)].isna().any(axis=1).sum())
        if not missing_required
        else None
    )

    return {
        "total_rows": int(len(df)),
        "total_columns": int(len(df.columns)),
        "columns": [str(column) for column in df.columns],
        "date_range": {"start": date_min, "end": date_max},
        "missing_values_by_column": missing,
        "rows_with_missing_required_values": required_missing_rows,
        "duplicate_date_rows": duplicate_count,
        "invalid_ohlc_rows": invalid_ohlc_count,
        "missing_required_columns": missing_required,
        "validation_status": (
            "passed"
            if not missing_required
            and not any(missing.get(column, 0) for column in _REQUIRED_COLUMNS)
            and duplicate_count == 0
            and invalid_ohlc_count == 0
            else "issues_found"
        ),
    }


def clean_ohlcv_data(
    file: UploadFile | None = None,
    *,
    path: str | Path | None = None,
    ext: str | None = None,
    data: bytes | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Run ingestion, normalization, validation, deduplication, and sorting.

    Returns the cleaned DataFrame and a quality report. The report describes
    the normalized data before duplicate removal, and includes final row count.
    """
    raw_df = load_ohlcv_file(file, path=path, ext=ext, data=data)
    df = normalize_columns(raw_df)

    missing_columns = validate_required_columns(df)
    if missing_columns:
        raise ValueError(
            "Missing required OHLCV columns: " + ", ".join(missing_columns)
        )

    df = convert_data_types(df)

    # Capture data-quality issues before any rows are removed.
    report = generate_quality_report(df)

    # Invalid candles should be rejected, not silently discarded.
    validate_ohlcv(df)

    duplicate_rows = int(df["Date"].duplicated(keep=False).sum())
    cleaned = remove_duplicate_candles(df, date_column="Date")
    cleaned = sort_candles(cleaned, date_column="Date")

    report["duplicate_date_rows"] = duplicate_rows
    report["duplicates_removed"] = int(len(df) - len(cleaned))
    report["cleaned_rows"] = int(len(cleaned))
    report["raw_rows"] = int(len(raw_df))
    report["validation_status"] = (
        "passed" if duplicate_rows == 0 and report["invalid_ohlc_rows"] == 0 else "issues_found"
    )

    return cleaned, report
