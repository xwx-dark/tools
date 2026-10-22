"""PDF and plain-text document operations."""

from io import BytesIO
from pathlib import Path
import re
from zipfile import ZIP_DEFLATED, ZipFile

import pymupdf

from toolhub.video import ProgressCallback, VideoProcessingError, VideoResult


class DocumentProcessingError(VideoProcessingError):
    """An actionable failure while reading or processing a document."""


def _open_pdf(data: bytes) -> pymupdf.Document:
    try:
        document = pymupdf.open(stream=data, filetype="pdf")
    except (pymupdf.FileDataError, RuntimeError, ValueError) as exc:
        raise DocumentProcessingError(
            "Could not read this PDF. Check that it is a valid, unencrypted PDF."
        ) from exc
    if document.needs_pass:
        document.close()
        raise DocumentProcessingError(
            "This PDF is password-protected. Remove its password before processing."
        )
    if document.page_count == 0:
        document.close()
        raise DocumentProcessingError("This PDF does not contain any pages.")
    return document


def get_pdf_page_count(data: bytes) -> int:
    """Return the page count for a readable PDF."""
    document = _open_pdf(data)
    try:
        return document.page_count
    finally:
        document.close()


def _zip_result(files: list[tuple[str, bytes]], filename: str) -> VideoResult:
    output = BytesIO()
    with ZipFile(output, mode="w", compression=ZIP_DEFLATED) as archive:
        for member_name, data in files:
            archive.writestr(member_name, data)
    return VideoResult(output.getvalue(), filename, "application/zip")


def merge_pdfs(
    inputs: list[tuple[bytes, str]], progress: ProgressCallback | None = None
) -> VideoResult:
    """Merge PDF files in upload order."""
    if len(inputs) < 2:
        raise DocumentProcessingError("Choose at least two PDF files to merge.")

    output = pymupdf.open()
    try:
        for index, (data, _) in enumerate(inputs):
            source = _open_pdf(data)
            try:
                output.insert_pdf(source)
            finally:
                source.close()
            if progress:
                progress((index + 1) / len(inputs))
        return VideoResult(
            output.tobytes(garbage=4, deflate=True),
            "merged.pdf",
            "application/pdf",
        )
    except (pymupdf.FileDataError, RuntimeError, ValueError) as exc:
        raise DocumentProcessingError(
            f"Could not merge the selected PDFs: {exc}"
        ) from exc
    finally:
        output.close()


def parse_page_selection(selection: str, page_count: int) -> list[int]:
    """Parse a comma-separated list of 1-based pages and inclusive ranges."""
    pages: list[int] = []
    try:
        for part in selection.split(","):
            token = part.strip()
            if not token:
                raise ValueError
            match = re.fullmatch(r"(\d+)(?:\s*-\s*(\d+))?", token)
            if not match:
                raise ValueError
            start = int(match.group(1))
            end = int(match.group(2) or start)
            if start < 1 or end < start or end > page_count:
                raise ValueError
            pages.extend(range(start, end + 1))
    except ValueError as exc:
        raise DocumentProcessingError(
            f"Enter valid page numbers from 1 to {page_count}, for example 1, 3-5."
        ) from exc
    if len(set(pages)) != len(pages):
        raise DocumentProcessingError("Do not select the same page more than once.")
    return pages


def split_pdf(
    data: bytes,
    input_name: str,
    page_selection: str,
    progress: ProgressCallback | None = None,
) -> VideoResult:
    """Export each selected page as its own PDF inside a ZIP archive."""
    source = _open_pdf(data)
    try:
        pages = parse_page_selection(page_selection, source.page_count)
        stem = Path(input_name or "document.pdf").stem or "document"
        outputs: list[tuple[str, bytes]] = []
        for index, page_number in enumerate(pages):
            page_document = pymupdf.open()
            try:
                page_document.insert_pdf(
                    source, from_page=page_number - 1, to_page=page_number - 1
                )
                outputs.append(
                    (
                        f"{stem}-page-{page_number}.pdf",
                        page_document.tobytes(garbage=4, deflate=True),
                    )
                )
            finally:
                page_document.close()
            if progress:
                progress((index + 1) / len(pages))
        return _zip_result(outputs, f"{stem}-pages.zip")
    finally:
        source.close()


def pdf_to_images(
    data: bytes,
    input_name: str,
    image_format: str,
    dpi: int,
    progress: ProgressCallback | None = None,
) -> VideoResult:
    """Render PDF pages as PNG or JPEG images in a ZIP archive."""
    normalized_format = image_format.upper()
    if normalized_format not in {"PNG", "JPEG"}:
        raise DocumentProcessingError("Choose PNG or JPEG as the image format.")
    if not 72 <= dpi <= 300:
        raise DocumentProcessingError(
            "Image resolution must be between 72 and 300 DPI."
        )

    source = _open_pdf(data)
    try:
        stem = Path(input_name or "document.pdf").stem or "document"
        extension = "jpg" if normalized_format == "JPEG" else "png"
        outputs = []
        matrix = pymupdf.Matrix(dpi / 72, dpi / 72)
        for index, page in enumerate(source):
            pixmap = page.get_pixmap(
                matrix=matrix, alpha=False, colorspace=pymupdf.csRGB
            )
            outputs.append(
                (
                    f"{stem}-page-{index + 1}.{extension}",
                    pixmap.tobytes(output="jpeg" if extension == "jpg" else "png"),
                )
            )
            if progress:
                progress((index + 1) / source.page_count)
        return _zip_result(outputs, f"{stem}-images.zip")
    except (pymupdf.FileDataError, RuntimeError, ValueError) as exc:
        raise DocumentProcessingError(f"Could not render PDF pages: {exc}") from exc
    finally:
        source.close()


def _wrap_text(text: str, max_width: float, font_size: float) -> list[str]:
    lines: list[str] = []
    for paragraph in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        if not paragraph:
            lines.append("")
            continue
        current = ""
        for word in paragraph.split():
            candidate = f"{current} {word}".strip()
            if (
                pymupdf.get_text_length(
                    candidate, fontname="helv", fontsize=font_size
                )
                <= max_width
            ):
                current = candidate
            else:
                if current:
                    lines.append(current)
                    current = ""
                fragment = ""
                for character in word:
                    candidate = fragment + character
                    if pymupdf.get_text_length(
                        candidate, fontname="helv", fontsize=font_size
                    ) <= max_width:
                        fragment = candidate
                    else:
                        lines.append(fragment)
                        fragment = character
                current = fragment
        if current:
            lines.append(current)
    return lines or [""]


def text_to_pdf(
    text: str,
    filename: str,
    progress: ProgressCallback | None = None,
) -> VideoResult:
    """Create a simple paginated PDF from a UTF-8 plain-text document."""
    page_width, page_height = 612, 792
    margin = 54
    font_size = 10
    line_height = 14
    max_width = page_width - 2 * margin
    lines = _wrap_text(text, max_width, font_size)
    lines_per_page = int((page_height - 2 * margin) // line_height)
    output = pymupdf.open()
    try:
        page_count = (len(lines) + lines_per_page - 1) // lines_per_page
        for page_index, start in enumerate(range(0, len(lines), lines_per_page)):
            page = output.new_page(width=page_width, height=page_height)
            for line_index, line in enumerate(lines[start : start + lines_per_page]):
                if line:
                    page.insert_text(
                        (margin, margin + (line_index + 1) * line_height),
                        line,
                        fontname="helv",
                        fontsize=font_size,
                    )
            if progress:
                progress((page_index + 1) / page_count)
        stem = Path(filename or "document.txt").stem or "document"
        return VideoResult(
            output.tobytes(garbage=4, deflate=True),
            f"{stem}.pdf",
            "application/pdf",
        )
    except (RuntimeError, ValueError) as exc:
        raise DocumentProcessingError(
            "Could not create a PDF from this text. Check for unsupported characters."
        ) from exc
    finally:
        output.close()


def pdf_to_text(
    data: bytes,
    input_name: str,
    progress: ProgressCallback | None = None,
) -> VideoResult:
    """Extract searchable text from every page of a PDF into a UTF-8 TXT file."""
    source = _open_pdf(data)
    try:
        page_text = []
        for index, page in enumerate(source):
            page_text.append(page.get_text("text"))
            if progress:
                progress((index + 1) / source.page_count)
        stem = Path(input_name or "document.pdf").stem or "document"
        return VideoResult(
            "\n\n".join(page_text).encode("utf-8"),
            f"{stem}.txt",
            "text/plain; charset=utf-8",
        )
    except (pymupdf.FileDataError, RuntimeError, ValueError) as exc:
        raise DocumentProcessingError(f"Could not extract PDF text: {exc}") from exc
    finally:
        source.close()
