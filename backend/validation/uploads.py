"""Validate analysis file uploads."""

from __future__ import annotations

import io
from typing import BinaryIO

from fastapi import UploadFile

from security.injection_scan import scan_chunk_for_ingestion

ALLOWED_EXTENSIONS = {"csv", "xls", "xlsx", "txt", "pdf"}
MACRO_MAGIC = (b"PK\x03\x04", b"\xd0\xcf\x11\xe0")  # xlsx/zip, xls OLE


class UploadValidationError(ValueError):
    pass


def validate_upload(file: UploadFile, *, max_bytes: int, max_pdf_pages: int) -> tuple[str, bytes]:
    filename = (file.filename or "").strip()
    if not filename or "." not in filename:
        raise UploadValidationError("Filename must include an extension.")
    ext = filename.rsplit(".", 1)[-1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise UploadValidationError(f"File type '{ext}' is not allowed.")

    raw = file.file.read()
    if len(raw) > max_bytes:
        raise UploadValidationError("File exceeds maximum allowed size.")
    if not raw:
        raise UploadValidationError("File is empty.")

    if ext in {"xls", "xlsx"} and raw.startswith(MACRO_MAGIC):
        # Reject macro-enabled legacy xls; xlsx is zip — allow but scan text extract later
        if ext == "xls" and b"Macros" in raw[:8192]:
            raise UploadValidationError("Macro-enabled workbooks are not allowed.")

    if ext == "pdf" and max_pdf_pages > 0:
        # Lightweight page count heuristic: count b"/Type /Page" occurrences
        pages = raw.count(b"/Type /Page")
        if pages and pages > max_pdf_pages:
            raise UploadValidationError("PDF exceeds maximum allowed pages.")

    text_probe = raw[:200_000].decode("utf-8", errors="ignore")
    finding = scan_chunk_for_ingestion("uploaded_file", text_probe)
    if finding.blocked:
        raise UploadValidationError("File content failed security screening.")

    return ext, raw


def buffer_for_pandas(data: bytes) -> BinaryIO:
    return io.BytesIO(data)
