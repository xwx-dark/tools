"""Page-based user interfaces for PDF and text document tools."""

from pathlib import Path

import streamlit as st

from toolhub.documents import (
    get_pdf_page_count,
    merge_pdfs,
    pdf_to_images,
    pdf_to_text,
    split_pdf,
    text_to_pdf,
)
from toolhub.video import VideoProcessingError, VideoResult
from toolhub.video_ui import (
    _poll_video_job,
    _process_button,
    _render_processing_progress,
    _request_back_to_category,
    _show_result,
)


DOCUMENT_TOOL_NAMES = (
    "PDF merger",
    "PDF splitter",
    "PDF to images",
    "Document converter",
)


def _upload_pdf(tool_key: str, *, multiple: bool = False):
    return st.file_uploader(
        "Choose PDF files" if multiple else "Choose a PDF",
        type=["pdf"],
        accept_multiple_files=multiple,
        key=f"documents-{tool_key}-upload",
        help="Document processing runs on the machine running this app.",
    )


def _render_merger() -> None:
    uploaded_files = _upload_pdf("merger", multiple=True)
    if not uploaded_files:
        return
    if len(uploaded_files) < 2:
        st.info("Choose at least two PDF files to merge.")
        return
    st.caption("PDFs will be combined in the order they appear above.")
    inputs = [(file.getvalue(), file.name) for file in uploaded_files]
    _process_button(
        "Merge PDFs",
        lambda progress: merge_pdfs(inputs, progress),
    )


def _render_splitter() -> None:
    uploaded_file = _upload_pdf("splitter")
    if uploaded_file is None:
        return
    input_data = uploaded_file.getvalue()
    try:
        page_count = get_pdf_page_count(input_data)
    except VideoProcessingError as exc:
        st.error(str(exc))
        return
    st.caption(f"This PDF contains {page_count} pages.")
    page_selection = st.text_input(
        "Pages to extract",
        value=f"1-{page_count}" if page_count > 1 else "1",
        key="documents-splitter-pages",
        help="Enter pages and inclusive ranges, separated by commas (for example: 1, 3-5).",
    )
    st.caption("Each selected page will be saved as a separate PDF in a ZIP file.")
    input_name = uploaded_file.name
    _process_button(
        "Split PDF",
        lambda progress: split_pdf(
            input_data, input_name, page_selection, progress
        ),
    )


def _render_pdf_to_images() -> None:
    uploaded_file = _upload_pdf("images")
    image_format = st.selectbox(
        "Image format", options=["PNG", "JPEG"], key="documents-images-format"
    )
    dpi = st.slider(
        "Resolution (DPI)",
        min_value=72,
        max_value=300,
        value=150,
        step=12,
        key="documents-images-dpi",
        help="Higher resolution creates sharper, larger images.",
    )
    if uploaded_file is None:
        return
    input_data = uploaded_file.getvalue()
    input_name = uploaded_file.name
    _process_button(
        "Convert PDF to images",
        lambda progress: pdf_to_images(
            input_data, input_name, image_format, dpi, progress
        ),
    )


def _render_converter() -> None:
    uploaded_file = st.file_uploader(
        "Choose a PDF or plain-text file",
        type=["pdf", "txt"],
        key="documents-converter-upload",
        help="Converts between searchable PDF text and UTF-8 plain text.",
    )
    if uploaded_file is None:
        return
    input_data = uploaded_file.getvalue()
    input_name = uploaded_file.name
    suffix = Path(input_name).suffix.casefold()
    if suffix == ".pdf":
        st.caption("Extract searchable text from the PDF into a UTF-8 .txt file.")
        _process_button(
            "Convert PDF to text",
            lambda progress: pdf_to_text(input_data, input_name, progress),
        )
    elif suffix == ".txt":
        st.caption(
            "Create a simple, paginated PDF. Plain text conversion does not "
            "preserve rich document layouts or formatting."
        )
        try:
            text = input_data.decode("utf-8-sig")
        except UnicodeDecodeError:
            st.error("The text file must use UTF-8 encoding.")
            return
        _process_button(
            "Convert text to PDF",
            lambda progress: text_to_pdf(text, input_name, progress),
        )
    else:
        st.error("Choose a PDF or a UTF-8 plain-text file.")


@st.fragment(run_every=0.25)
def render_document_tool_page() -> None:
    _poll_video_job()
    if st.button("← Back to Documents", key="documents-page-back"):
        _request_back_to_category()
        st.rerun()

    tool_name = st.session_state.get("active_tool")
    st.title(f"📄  {tool_name or 'Document tool'}")
    st.divider()
    st.caption("Document processing runs on the machine running this app.")
    _render_processing_progress()

    tools = {
        "PDF merger": _render_merger,
        "PDF splitter": _render_splitter,
        "PDF to images": _render_pdf_to_images,
        "Document converter": _render_converter,
    }
    render_tool = tools.get(tool_name)
    if render_tool is None:
        st.error("This document tool is not available.")
        return
    render_tool()

    result = st.session_state.get("video_tool_result")
    if (
        isinstance(result, VideoResult)
        and st.session_state.get("video_processing_tool") == tool_name
    ):
        st.divider()
        _show_result(result)
