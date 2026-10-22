"""Operations for uploaded files and archives."""

import csv
from hashlib import sha256
from io import BytesIO, StringIO
from pathlib import PurePosixPath
import tarfile
from zipfile import ZIP_DEFLATED, BadZipFile, ZipFile

from toolhub.video import ProgressCallback, VideoProcessingError, VideoResult


class FileToolError(VideoProcessingError):
    """An actionable failure while processing uploaded files."""


def _safe_basename(filename: str) -> str:
    name = PurePosixPath(filename.replace("\\", "/")).name
    if name in {"", ".", ".."}:
        return "file"
    return name


def _unique_name(filename: str, used_names: set[str]) -> str:
    path = PurePosixPath(_safe_basename(filename))
    stem, suffix = path.stem, path.suffix
    candidate = path.name
    counter = 2
    while candidate.casefold() in used_names:
        candidate = f"{stem}_{counter}{suffix}"
        counter += 1
    used_names.add(candidate.casefold())
    return candidate


def _csv_safe(value: str) -> str:
    if value.startswith(("=", "+", "-", "@", "\t", "\r")):
        return "'" + value
    return value


def _csv_result(rows: list[dict[str, object]], filename: str) -> VideoResult:
    output = StringIO(newline="")
    if rows:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: _csv_safe(str(value)) if isinstance(value, str) else value
                    for key, value in row.items()
                }
            )
    return VideoResult(
        output.getvalue().encode("utf-8-sig"),
        filename,
        "text/csv; charset=utf-8",
    )


def rename_files(
    inputs: list[tuple[bytes, str]],
    prefix: str = "",
    suffix: str = "",
    start_number: int = 1,
    number_width: int = 2,
    progress: ProgressCallback | None = None,
) -> VideoResult:
    """Add a prefix, sequence number, and suffix while preserving extensions."""
    if not inputs:
        raise FileToolError("Choose at least one file to rename.")
    if start_number < 0 or not 1 <= number_width <= 8:
        raise FileToolError("Use a non-negative start number and width from 1 to 8.")

    output = BytesIO()
    used_names: set[str] = set()
    with ZipFile(output, mode="w", compression=ZIP_DEFLATED) as archive:
        for index, (data, original_name) in enumerate(inputs):
            original = PurePosixPath(_safe_basename(original_name))
            sequence = str(start_number + index).zfill(number_width)
            renamed = _unique_name(
                f"{prefix}{sequence}_{original.stem}{suffix}{original.suffix}",
                used_names,
            )
            archive.writestr(renamed, data)
            if progress:
                progress((index + 1) / len(inputs))
    return VideoResult(output.getvalue(), "renamed-files.zip", "application/zip")


def create_zip(
    inputs: list[tuple[bytes, str]], progress: ProgressCallback | None = None
) -> VideoResult:
    """Create a ZIP archive from uploaded files."""
    if not inputs:
        raise FileToolError("Choose at least one file to archive.")

    output = BytesIO()
    used_names: set[str] = set()
    with ZipFile(output, mode="w", compression=ZIP_DEFLATED) as archive:
        for index, (data, filename) in enumerate(inputs):
            archive.writestr(_unique_name(filename, used_names), data)
            if progress:
                progress((index + 1) / len(inputs))
    return VideoResult(output.getvalue(), "files.zip", "application/zip")


def inspect_archive(
    data: bytes,
    filename: str,
    progress: ProgressCallback | None = None,
) -> VideoResult:
    """List ZIP or TAR archive entries as a CSV without extracting them."""
    entries: list[tuple[str, int, int | None, str]] = []
    normalized_name = filename.casefold()
    if normalized_name.endswith(".zip"):
        try:
            with ZipFile(BytesIO(data)) as archive:
                for item in archive.infolist():
                    entries.append(
                        (
                            item.filename,
                            item.file_size,
                            item.compress_size,
                            "directory" if item.is_dir() else "file",
                        )
                    )
        except (BadZipFile, OSError, EOFError) as exc:
            raise FileToolError("Could not read this ZIP archive.") from exc
    elif normalized_name.endswith((".tar", ".tar.gz", ".tgz", ".tar.bz2", ".tar.xz")):
        try:
            with tarfile.open(fileobj=BytesIO(data), mode="r:*") as archive:
                for item in archive.getmembers():
                    item_type = (
                        "directory"
                        if item.isdir()
                        else "file"
                        if item.isfile()
                        else "link or special"
                    )
                    entries.append((item.name, item.size, None, item_type))
        except (tarfile.TarError, OSError, EOFError) as exc:
            raise FileToolError("Could not read this TAR archive.") from exc
    else:
        raise FileToolError(
            "Choose a ZIP, TAR, TAR.GZ, TGZ, TAR.BZ2, or TAR.XZ archive."
        )

    if not entries:
        raise FileToolError("This archive does not contain any entries.")
    rows = []
    for index, (name, size, compressed_size, entry_type) in enumerate(entries):
        rows.append(
            {
                "name": name,
                "type": entry_type,
                "size_bytes": size,
                "compressed_size_bytes": (
                    "" if compressed_size is None else compressed_size
                ),
            }
        )
        if progress:
            progress((index + 1) / len(entries))

    report_name = _safe_basename(filename) + "-contents.csv"
    return _csv_result(rows, report_name)


def find_duplicates(
    inputs: list[tuple[bytes, str]], progress: ProgressCallback | None = None
) -> VideoResult:
    """Report uploaded files with matching SHA-256 content hashes."""
    if len(inputs) < 2:
        raise FileToolError("Choose at least two files to find duplicates.")

    hashes = [sha256(data).hexdigest() for data, _ in inputs]
    groups: dict[str, list[int]] = {}
    for index, digest in enumerate(hashes):
        groups.setdefault(digest, []).append(index)

    duplicate_groups = {
        digest: members
        for digest, members in groups.items()
        if len(members) > 1
    }
    group_ids = {
        index: group_number
        for group_number, members in enumerate(duplicate_groups.values(), start=1)
        for index in members
    }
    rows: list[dict[str, object]] = []
    for index, ((data, filename), digest) in enumerate(zip(inputs, hashes)):
        rows.append(
            {
                "name": _safe_basename(filename),
                "size_bytes": len(data),
                "sha256": digest,
                "duplicate_group": group_ids.get(index, ""),
            }
        )
        if progress:
            progress((index + 1) / len(inputs))
    return _csv_result(rows, "duplicate-report.csv")


def analyze_file_sizes(
    inputs: list[tuple[bytes, str]], progress: ProgressCallback | None = None
) -> VideoResult:
    """Create a largest-first CSV inventory of uploaded files."""
    if not inputs:
        raise FileToolError("Choose at least one file to analyze.")

    sorted_inputs = sorted(inputs, key=lambda item: len(item[0]), reverse=True)
    rows: list[dict[str, object]] = []
    for index, (data, filename) in enumerate(sorted_inputs):
        clean_name = _safe_basename(filename)
        rows.append(
            {
                "name": clean_name,
                "extension": PurePosixPath(clean_name).suffix.casefold(),
                "size_bytes": len(data),
            }
        )
        if progress:
            progress((index + 1) / len(sorted_inputs))
    return _csv_result(rows, "file-size-report.csv")
