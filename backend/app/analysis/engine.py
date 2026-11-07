"""Orchestrates OHLCV cleaning and future swing/intraday analysis."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import UploadFile

from analysis.data_cleaning import (
    clean_ohlcv_data,
    convert_data_types,
    generate_quality_report,
    normalize_columns,
    remove_duplicate_candles,
    sort_candles,
    validate_ohlcv,
    validate_required_columns,
)


class OHLCVAnalysisEngine:
    """Coordinates data cleaning and strategy modules for OHLCV workflows."""

    def __init__(self) -> None:
        self._swing_handlers: dict[str, Any] = {}
        self._intraday_handlers: dict[str, Any] = {}

    def analyze(
        self,
        df: pd.DataFrame | None = None,
        *,
        file: UploadFile | None = None,
        path: str | Path | None = None,
        ext: str | None = None,
        data: bytes | None = None,
        data_type: str | None = None,
    ) -> dict[str, Any]:
        """Accept OHLCV input, run cleaning, and return a pipeline result envelope."""
        if df is not None:
            cleaned, quality_report = self._clean_dataframe(df)
        elif file is not None or path is not None or data is not None:
            cleaned, quality_report = clean_ohlcv_data(
                file,
                path=path,
                ext=ext,
                data=data,
            )
        else:
            raise ValueError("Provide OHLCV data as a DataFrame or file input.")

        return {
            "data_type": data_type,
            "quality_report": quality_report,
            "row_count": int(len(cleaned)),
            "swing": self._run_swing(cleaned, data_type=data_type),
            "intraday": self._run_intraday(cleaned, data_type=data_type),
        }

    def _clean_dataframe(self, df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
        """Apply the same cleaning stages as ``clean_ohlcv_data`` on an in-memory frame."""
        normalized = normalize_columns(df)

        missing_columns = validate_required_columns(normalized)
        if missing_columns:
            raise ValueError(
                "Missing required OHLCV columns: " + ", ".join(missing_columns)
            )

        typed = convert_data_types(normalized)
        report = generate_quality_report(typed)
        validate_ohlcv(typed)

        duplicate_rows = int(typed["Date"].duplicated(keep=False).sum())
        cleaned = remove_duplicate_candles(typed, date_column="Date")
        cleaned = sort_candles(cleaned, date_column="Date")

        report["duplicate_date_rows"] = duplicate_rows
        report["duplicates_removed"] = int(len(typed) - len(cleaned))
        report["cleaned_rows"] = int(len(cleaned))
        report["raw_rows"] = int(len(df))
        report["validation_status"] = (
            "passed"
            if duplicate_rows == 0 and report.get("invalid_ohlc_rows", 0) == 0
            else "issues_found"
        )

        return cleaned, report

    def _run_swing(
        self,
        df: pd.DataFrame,
        *,
        data_type: str | None,
    ) -> dict[str, Any]:
        """Placeholder for swing strategy modules."""
        _ = (df, data_type, self._swing_handlers)
        return {}

    def _run_intraday(
        self,
        df: pd.DataFrame,
        *,
        data_type: str | None,
    ) -> dict[str, Any]:
        """Placeholder for intraday strategy modules."""
        _ = (df, data_type, self._intraday_handlers)
        return {}
