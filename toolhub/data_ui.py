"""Page-based user interfaces for CSV and JSON tools."""

import csv
import json
from io import StringIO
from pathlib import Path
import tempfile

import streamlit as st

from make_vlm_postman_request import build_request_body
from toolhub.data import convert_table, csv_to_json, format_json, preview_csv
from toolhub.video import VideoResult
from toolhub.video_ui import (
    _poll_video_job,
    _process_button,
    _render_processing_progress,
    _request_back_to_category,
    _show_result,
)


DATA_TOOL_NAMES = (
    "CSV viewer",
    "JSON formatter",
    "CSV to JSON",
    "Data table converter",
    "Convert image to VLM request",
)

_DELIMITERS = {
    "Comma (,)": ",",
    "Semicolon (;)": ";",
    "Tab": "\t",
    "Pipe (|)": "|",
}


def _render_csv_viewer() -> None:
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=["csv", "txt"],
        key="data-csv-viewer-upload",
        help="CSV data must be UTF-8 text with a header row.",
    )
    delimiter_label = st.selectbox(
        "Delimiter", list(_DELIMITERS), key="data-csv-viewer-delimiter"
    )
    sample_rows = st.slider(
        "Rows to preview",
        min_value=10,
        max_value=500,
        value=100,
        step=10,
        key="data-csv-viewer-rows",
    )
    if uploaded_file is None:
        return
    data, filename = uploaded_file.getvalue(), uploaded_file.name
    delimiter = _DELIMITERS[delimiter_label]
    _process_button(
        "Prepare CSV preview",
        lambda progress: preview_csv(
            data, filename, delimiter, sample_rows, progress
        ),
    )


def _render_json_formatter() -> None:
    uploaded_file = st.file_uploader(
        "Choose a JSON file",
        type=["json", "txt"],
        key="data-json-formatter-upload",
        help="JSON data must be UTF-8 encoded.",
    )
    if uploaded_file is None:
        return
    data, filename = uploaded_file.getvalue(), uploaded_file.name
    _process_button(
        "Format JSON",
        lambda progress: format_json(data, filename, progress),
    )


def _render_csv_to_json() -> None:
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=["csv", "txt"],
        key="data-csv-to-json-upload",
        help="CSV data must have a header row; values remain strings in JSON.",
    )
    delimiter_label = st.selectbox(
        "Delimiter", list(_DELIMITERS), key="data-csv-to-json-delimiter"
    )
    if uploaded_file is None:
        return
    data, filename = uploaded_file.getvalue(), uploaded_file.name
    delimiter = _DELIMITERS[delimiter_label]
    _process_button(
        "Convert CSV to JSON",
        lambda progress: csv_to_json(data, filename, delimiter, progress),
    )


def _render_table_converter() -> None:
    input_format = st.radio(
        "Input format",
        options=["CSV", "TSV"],
        horizontal=True,
        key="data-table-input-format",
    )
    uploaded_file = st.file_uploader(
        f"Choose a {input_format} file",
        type=["csv", "tsv", "txt"],
        key="data-table-converter-upload",
        help="Input data must have a header row and use UTF-8 encoding.",
    )
    if uploaded_file is None:
        return
    data, filename = uploaded_file.getvalue(), uploaded_file.name
    st.caption(f"Output format: {'TSV' if input_format == 'CSV' else 'CSV'}")
    _process_button(
        f"Convert {input_format} to {'TSV' if input_format == 'CSV' else 'CSV'}",
        lambda progress: convert_table(data, filename, input_format, progress),
    )


def _render_image_vlm_request() -> None:
    uploaded_file = st.file_uploader(
        "Choose an image",
        type=["bmp", "jpeg", "jpg", "png", "tif", "tiff", "webp"],
        key="data-vlm-request-upload",
        help="The generated JSON contains a JPEG-encoded image as a base64 string.",
    )
    if uploaded_file is None:
        return

    data = uploaded_file.getvalue()
    filename = uploaded_file.name or "image.jpg"

    def create_request(progress) -> VideoResult:
        with tempfile.TemporaryDirectory(
            prefix="everyday-tools-vlm-request-"
        ) as directory:
            image_path = Path(directory) / (
                f"image{Path(filename).suffix or '.jpg'}"
            )
            image_path.write_bytes(data)
            body = build_request_body(image_path)
        progress(1.0)
        request_json = json.dumps(body, indent=2) + "\n"
        return VideoResult(
            request_json.encode("utf-8"),
            f"{Path(filename).stem or 'image'}-vlm-request.json",
            "application/json",
        )

    _process_button("Convert image to VLM request", create_request)


def _show_data_result(tool_name: str, result: VideoResult) -> None:
    if tool_name == "CSV viewer":
        rows = list(
            csv.DictReader(
                StringIO(result.data.decode("utf-8"), newline=""),
                delimiter=",",
            )
        )
        st.caption(f"Showing {len(rows)} preview rows.")
        st.dataframe(rows, hide_index=True, use_container_width=True)
    elif tool_name == "JSON formatter":
        st.code(result.data.decode("utf-8"), language="json")
    _show_result(result)


@st.fragment(run_every=0.25)
def render_data_tool_page() -> None:
    _poll_video_job()
    if st.button("← Back to Data", key="data-page-back"):
        _request_back_to_category()
        st.rerun()

    tool_name = st.session_state.get("active_tool")
    st.title(f"📊  {tool_name or 'Data tool'}")
    st.divider()
    st.caption("Data processing runs on the machine running this app.")
    _render_processing_progress()

    tools = {
        "CSV viewer": _render_csv_viewer,
        "JSON formatter": _render_json_formatter,
        "CSV to JSON": _render_csv_to_json,
        "Data table converter": _render_table_converter,
        "Convert image to VLM request": _render_image_vlm_request,
    }
    render_tool = tools.get(tool_name)
    if render_tool is None:
        st.error("This data tool is not available.")
        return
    render_tool()

    result = st.session_state.get("video_tool_result")
    if (
        isinstance(result, VideoResult)
        and st.session_state.get("video_processing_tool") == tool_name
    ):
        st.divider()
        _show_data_result(tool_name, result)
