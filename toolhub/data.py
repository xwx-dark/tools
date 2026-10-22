"""CSV and JSON operations for the Data category."""

import csv
from io import StringIO
from itertools import islice
import json
from pathlib import Path
from typing import Any

from toolhub.video import ProgressCallback, VideoProcessingError, VideoResult


class DataProcessingError(VideoProcessingError):
    """An actionable failure while reading or converting structured data."""


def _decode_text(data: bytes, filename: str) -> str:
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise DataProcessingError(
            f"{filename or 'The uploaded file'} must use UTF-8 encoding."
        ) from exc


def _parse_table(text: str, delimiter: str, filename: str) -> tuple[list[str], list[list[str]]]:
    if delimiter not in {",", ";", "\t", "|"}:
        raise DataProcessingError("Choose comma, semicolon, tab, or pipe as the delimiter.")
    try:
        reader = csv.reader(StringIO(text, newline=""), delimiter=delimiter, strict=True)
        rows = list(reader)
    except csv.Error as exc:
        raise DataProcessingError(f"Could not parse {filename or 'this table'}: {exc}") from exc
    if not rows:
        raise DataProcessingError("The uploaded table is empty.")

    headers = _validate_headers(rows[0])
    records = rows[1:]
    for row_number, row in enumerate(records, start=2):
        if len(row) != len(headers):
            raise DataProcessingError(
                f"Row {row_number} has {len(row)} values; expected {len(headers)}."
            )
    return headers, records


def _validate_headers(headers: list[str]) -> list[str]:
    if not headers or all(not header.strip() for header in headers):
        raise DataProcessingError("The table must have a non-empty header row.")
    normalized_headers = [header.strip() for header in headers]
    if any(not header for header in normalized_headers):
        raise DataProcessingError("Every column must have a header.")
    if len({header.casefold() for header in normalized_headers}) != len(normalized_headers):
        raise DataProcessingError("Column names must be unique.")
    return normalized_headers


def _table_to_text(headers: list[str], rows: list[list[str]], delimiter: str) -> str:
    output = StringIO(newline="")
    writer = csv.writer(output, delimiter=delimiter, lineterminator="\r\n")
    writer.writerow(headers)
    writer.writerows(rows)
    return output.getvalue()


def preview_csv(
    data: bytes,
    filename: str,
    delimiter: str,
    sample_rows: int,
    progress: ProgressCallback | None = None,
) -> VideoResult:
    """Produce a downloadable CSV sample with its header and selected rows."""
    if not 1 <= sample_rows <= 500:
        raise DataProcessingError("Choose a preview size from 1 to 500 rows.")
    if delimiter not in {",", ";", "\t", "|"}:
        raise DataProcessingError("Choose comma, semicolon, tab, or pipe as the delimiter.")
    reader = csv.reader(
        StringIO(_decode_text(data, filename), newline=""),
        delimiter=delimiter,
        strict=True,
    )
    try:
        raw_headers = next(reader)
        headers = _validate_headers(raw_headers)
        sample = list(islice(reader, sample_rows))
    except StopIteration as exc:
        raise DataProcessingError("The uploaded table is empty.") from exc
    except csv.Error as exc:
        raise DataProcessingError(
            f"Could not parse {filename or 'this table'}: {exc}"
        ) from exc
    for row_number, row in enumerate(sample, start=2):
        if len(row) != len(headers):
            raise DataProcessingError(
                f"Row {row_number} has {len(row)} values; expected {len(headers)}."
            )
    if progress:
        progress(1.0)
    stem = Path(filename or "data.csv").stem or "data"
    return VideoResult(
        _table_to_text(headers, sample, ",").encode("utf-8"),
        f"{stem}-preview.csv",
        "text/csv; charset=utf-8",
    )


def format_json(
    data: bytes,
    filename: str,
    progress: ProgressCallback | None = None,
) -> VideoResult:
    """Validate JSON and return a consistently indented UTF-8 document."""
    text = _decode_text(data, filename)

    def reject_nonstandard_constant(value: str) -> None:
        raise ValueError(f"{value} is not valid JSON.")

    try:
        value: Any = json.loads(text, parse_constant=reject_nonstandard_constant)
    except json.JSONDecodeError as exc:
        raise DataProcessingError(
            f"Invalid JSON at line {exc.lineno}, column {exc.colno}: {exc.msg}."
        ) from exc
    except ValueError as exc:
        raise DataProcessingError(f"Invalid JSON: {exc}") from exc
    if progress:
        progress(1.0)
    stem = Path(filename or "data.json").stem or "data"
    return VideoResult(
        (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8"),
        f"{stem}-formatted.json",
        "application/json",
    )


def csv_to_json(
    data: bytes,
    filename: str,
    delimiter: str,
    progress: ProgressCallback | None = None,
) -> VideoResult:
    """Convert a header-based CSV table into a JSON array of records."""
    headers, rows = _parse_table(_decode_text(data, filename), delimiter, filename)
    records = [dict(zip(headers, row)) for row in rows]
    if progress:
        progress(1.0)
    stem = Path(filename or "data.csv").stem or "data"
    return VideoResult(
        (json.dumps(records, ensure_ascii=False, indent=2) + "\n").encode("utf-8"),
        f"{stem}.json",
        "application/json",
    )


def convert_table(
    data: bytes,
    filename: str,
    input_format: str,
    progress: ProgressCallback | None = None,
) -> VideoResult:
    """Convert a header-based CSV or TSV table into the other format."""
    normalized_format = input_format.upper()
    delimiters = {"CSV": ",", "TSV": "\t"}
    if normalized_format not in delimiters:
        raise DataProcessingError("Choose CSV or TSV as the input format.")
    input_delimiter = delimiters[normalized_format]
    output_format = "TSV" if normalized_format == "CSV" else "CSV"
    output_delimiter = delimiters[output_format]
    headers, rows = _parse_table(
        _decode_text(data, filename), input_delimiter, filename
    )
    output = _table_to_text(headers, rows, output_delimiter)
    if progress:
        progress(1.0)
    stem = Path(filename or f"data.{normalized_format.casefold()}").stem or "data"
    extension = output_format.casefold()
    mime_type = "text/tab-separated-values; charset=utf-8" if extension == "tsv" else "text/csv; charset=utf-8"
    return VideoResult(
        output.encode("utf-8"),
        f"{stem}.{extension}",
        mime_type,
    )
