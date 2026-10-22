"""Page-based user interfaces for developer utility tools."""

import streamlit as st

from toolhub.developer import (
    base64_decode,
    base64_encode,
    generate_hash,
    url_decode,
    url_encode,
    validate_json,
)
from toolhub.video import VideoResult
from toolhub.video_ui import (
    _poll_video_job,
    _process_button,
    _render_processing_progress,
    _request_back_to_category,
    _show_result,
)


DEVELOPER_TOOL_NAMES = (
    "JSON validator",
    "Base64 encoder",
    "URL encoder",
    "Hash generator",
)


def _render_json_validator() -> None:
    uploaded_file = st.file_uploader(
        "Choose a JSON file",
        type=["json", "txt"],
        key="developer-json-upload",
        help="The JSON file must use UTF-8 encoding.",
    )
    if uploaded_file is None:
        return
    data, filename = uploaded_file.getvalue(), uploaded_file.name
    _process_button(
        "Validate JSON",
        lambda progress: validate_json(data, filename, progress),
    )


def _render_base64() -> None:
    operation = st.radio(
        "Operation",
        options=["Encode text", "Encode file", "Decode Base64 text", "Decode Base64 file"],
        key="developer-base64-operation",
    )
    if operation in {"Encode text", "Decode Base64 text"}:
        text = st.text_area(
            "Text to encode" if operation == "Encode text" else "Base64 text to decode",
            key="developer-base64-text",
        )
        data = text.encode("utf-8")
        if operation == "Encode text":
            _process_button(
                "Encode as Base64",
                lambda progress: base64_encode(data, "text.txt", progress),
            )
        else:
            _process_button(
                "Decode Base64",
                lambda progress: base64_decode(data, "text.b64", progress),
            )
        return

    uploaded_file = st.file_uploader(
        "Choose a file",
        key=f"developer-base64-{operation.casefold().replace(' ', '-')}-upload",
        help="File contents are handled as bytes.",
    )
    if uploaded_file is None:
        return
    data, filename = uploaded_file.getvalue(), uploaded_file.name
    if operation == "Encode file":
        _process_button(
            "Encode as Base64",
            lambda progress: base64_encode(data, filename, progress),
        )
    else:
        _process_button(
            "Decode Base64",
            lambda progress: base64_decode(data, filename, progress),
        )


def _render_url_encoder() -> None:
    operation = st.radio(
        "Operation",
        options=["Encode component", "Decode component"],
        horizontal=True,
        key="developer-url-operation",
        help="Processes a URL component, not a complete URL.",
    )
    text = st.text_area(
        "Text to encode" if operation == "Encode component" else "Encoded component",
        key="developer-url-text",
    )
    data = text.encode("utf-8")
    if operation == "Encode component":
        _process_button(
            "Encode URL component",
            lambda progress: url_encode(data, progress),
        )
    else:
        _process_button(
            "Decode URL component",
            lambda progress: url_decode(data, progress),
        )


def _render_hash_generator() -> None:
    algorithm = st.selectbox(
        "Hash algorithm",
        options=["SHA-256", "SHA-512", "SHA-1", "MD5"],
        key="developer-hash-algorithm",
        help="MD5 and SHA-1 are included only for compatibility checks.",
    )
    source = st.radio(
        "Input type",
        options=["Text", "File"],
        horizontal=True,
        key="developer-hash-source",
    )
    if source == "Text":
        text = st.text_area("Text to hash", key="developer-hash-text")
        data = text.encode("utf-8")
    else:
        uploaded_file = st.file_uploader(
            "Choose a file",
            key="developer-hash-upload",
            help="The checksum is computed from the file's exact bytes.",
        )
        if uploaded_file is None:
            return
        data = uploaded_file.getvalue()
    _process_button(
        "Generate checksum",
        lambda progress: generate_hash(data, algorithm, progress),
    )


def _show_developer_result(tool_name: str, result: VideoResult) -> None:
    if tool_name == "JSON validator":
        st.success("Valid JSON.")
        st.code(result.data.decode("utf-8"), language="json")
    elif tool_name == "Base64 encoder":
        if result.mime_type.startswith("text/"):
            st.code(result.data.decode("ascii"), language="text")
        else:
            st.info("Decoded output is binary. Download the file to use its contents.")
    elif tool_name == "URL encoder":
        st.code(result.data.decode("utf-8"), language="text")
    elif tool_name == "Hash generator":
        st.code(result.data.decode("ascii"), language="text")
    _show_result(result)


@st.fragment(run_every=0.25)
def render_developer_tool_page() -> None:
    _poll_video_job()
    if st.button("← Back to Developer", key="developer-page-back"):
        _request_back_to_category()
        st.rerun()

    tool_name = st.session_state.get("active_tool")
    st.title(f"💻  {tool_name or 'Developer tool'}")
    st.divider()
    st.caption("Developer utility processing runs on the machine running this app.")
    _render_processing_progress()

    tools = {
        "JSON validator": _render_json_validator,
        "Base64 encoder": _render_base64,
        "URL encoder": _render_url_encoder,
        "Hash generator": _render_hash_generator,
    }
    render_tool = tools.get(tool_name)
    if render_tool is None:
        st.error("This developer tool is not available.")
        return
    render_tool()

    result = st.session_state.get("video_tool_result")
    if (
        isinstance(result, VideoResult)
        and st.session_state.get("video_processing_tool") == tool_name
    ):
        st.divider()
        _show_developer_result(tool_name, result)
