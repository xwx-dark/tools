"""Developer utilities for JSON, Base64, URLs, and checksums."""

import base64
import binascii
import hashlib
import json
from pathlib import Path
import re
from urllib.parse import quote, unquote_to_bytes
from typing import Any

from toolhub.video import ProgressCallback, VideoProcessingError, VideoResult


class DeveloperToolError(VideoProcessingError):
    """An actionable failure while processing developer utility input."""


def _decode_utf8(data: bytes, description: str) -> str:
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise DeveloperToolError(f"{description} must be UTF-8 text.") from exc


def _complete(progress: ProgressCallback | None) -> None:
    if progress:
        progress(1.0)


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"{value} is not permitted by the JSON specification.")


def validate_json(
    data: bytes,
    filename: str = "data.json",
    progress: ProgressCallback | None = None,
) -> VideoResult:
    """Validate strict JSON and return a normalized, readable JSON document."""
    text = _decode_utf8(data, "JSON input")
    try:
        value: Any = json.loads(text, parse_constant=_reject_json_constant)
    except json.JSONDecodeError as exc:
        raise DeveloperToolError(
            f"Invalid JSON at line {exc.lineno}, column {exc.colno}: {exc.msg}."
        ) from exc
    except ValueError as exc:
        raise DeveloperToolError(f"Invalid JSON: {exc}") from exc
    _complete(progress)
    stem = Path(filename or "data.json").stem or "data"
    try:
        formatted = json.dumps(
            value, ensure_ascii=False, indent=2, allow_nan=False
        ) + "\n"
    except ValueError as exc:
        raise DeveloperToolError(f"Invalid JSON: {exc}") from exc
    return VideoResult(
        formatted.encode("utf-8"),
        f"{stem}-validated.json",
        "application/json",
    )


def base64_encode(
    data: bytes,
    filename: str = "data.bin",
    progress: ProgressCallback | None = None,
) -> VideoResult:
    """Encode bytes as Base64 ASCII."""
    _complete(progress)
    stem = Path(filename or "data.bin").stem or "data"
    encoded = base64.b64encode(data)
    return VideoResult(encoded, f"{stem}.b64", "text/plain; charset=us-ascii")


def base64_decode(
    data: bytes,
    filename: str = "data.b64",
    progress: ProgressCallback | None = None,
) -> VideoResult:
    """Strictly decode Base64 text into its original bytes."""
    text = _decode_utf8(data, "Base64 input")
    compact = "".join(text.split())
    try:
        decoded = base64.b64decode(compact, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise DeveloperToolError(
            "Invalid Base64 input. Check its characters and padding."
        ) from exc
    _complete(progress)
    stem = Path(filename or "data.b64").stem or "data"
    return VideoResult(decoded, f"{stem}-decoded.bin", "application/octet-stream")


def url_encode(
    data: bytes,
    progress: ProgressCallback | None = None,
) -> VideoResult:
    """Percent-encode UTF-8 text as a URL component."""
    text = _decode_utf8(data, "URL input")
    encoded = quote(text, safe="", encoding="utf-8", errors="strict")
    _complete(progress)
    return VideoResult(
        encoded.encode("ascii"), "url-encoded.txt", "text/plain; charset=us-ascii"
    )


def url_decode(
    data: bytes,
    progress: ProgressCallback | None = None,
) -> VideoResult:
    """Decode percent escapes as a UTF-8 URL component."""
    text = _decode_utf8(data, "URL input")
    if re.search(r"%(?![0-9a-fA-F]{2})", text):
        raise DeveloperToolError(
            "Invalid URL encoding: every percent sign must be followed by two hex digits."
        )
    try:
        decoded = unquote_to_bytes(text).decode("utf-8")
    except UnicodeDecodeError as exc:
        raise DeveloperToolError(
            "The decoded URL component is not valid UTF-8."
        ) from exc
    _complete(progress)
    return VideoResult(
        decoded.encode("utf-8"), "url-decoded.txt", "text/plain; charset=utf-8"
    )


def generate_hash(
    data: bytes,
    algorithm: str,
    progress: ProgressCallback | None = None,
) -> VideoResult:
    """Generate a checksum for bytes using a supported hashlib algorithm."""
    normalized_algorithm = algorithm.casefold().replace("-", "")
    if normalized_algorithm not in {"md5", "sha1", "sha256", "sha512"}:
        raise DeveloperToolError("Choose MD5, SHA-1, SHA-256, or SHA-512.")
    try:
        digest = hashlib.new(normalized_algorithm, data).hexdigest()
    except (ValueError, TypeError) as exc:
        raise DeveloperToolError(
            f"Could not generate a {algorithm} checksum: {exc}"
        ) from exc
    _complete(progress)
    result = f"{normalized_algorithm.upper()}: {digest}\n"
    return VideoResult(
        result.encode("ascii"),
        f"{normalized_algorithm}-checksum.txt",
        "text/plain; charset=us-ascii",
    )
