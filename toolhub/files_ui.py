"""Page-based user interfaces for uploaded-file tools."""

import streamlit as st

from toolhub.files import (
    analyze_file_sizes,
    create_zip,
    find_duplicates,
    inspect_archive,
    rename_files,
)
from toolhub.video import VideoResult
from toolhub.video_ui import (
    _poll_video_job,
    _process_button,
    _render_processing_progress,
    _request_back_to_category,
    _show_result,
)


FILE_TOOL_NAMES = (
    "File renamer",
    "Archive manager",
    "Duplicate finder",
    "File size analyzer",
)


def _upload_files(tool_key: str, *, multiple: bool = True):
    return st.file_uploader(
        "Choose files",
        key=f"files-{tool_key}-upload",
        accept_multiple_files=multiple,
        help="File tools operate only on files uploaded to this page.",
    )


def _file_inputs(uploaded_files) -> list[tuple[bytes, str]]:
    return [
        (uploaded_file.getvalue(), uploaded_file.name)
        for uploaded_file in uploaded_files
    ]


def _render_renamer() -> None:
    uploaded_files = _upload_files("renamer")
    prefix = st.text_input("Prefix", key="files-renamer-prefix")
    suffix = st.text_input("Suffix", key="files-renamer-suffix")
    first_number = st.number_input(
        "Start numbering at",
        min_value=0,
        max_value=999999,
        value=1,
        step=1,
        key="files-renamer-start",
    )
    number_width = st.slider(
        "Number width",
        min_value=1,
        max_value=8,
        value=2,
        key="files-renamer-width",
        help="Numbers are zero-padded to this width.",
    )
    if not uploaded_files:
        return
    inputs = _file_inputs(uploaded_files)
    _process_button(
        "Rename files",
        lambda progress: rename_files(
            inputs, prefix, suffix, first_number, number_width, progress
        ),
    )


def _render_archive_manager() -> None:
    mode = st.radio(
        "Archive action",
        options=["Create ZIP", "Inspect archive"],
        horizontal=True,
        key="files-archive-mode",
    )
    if mode == "Create ZIP":
        uploaded_files = _upload_files("archive-create")
        if not uploaded_files:
            return
        st.caption("Files are stored by filename at the archive root.")
        inputs = _file_inputs(uploaded_files)
        _process_button(
            "Create ZIP archive",
            lambda progress: create_zip(inputs, progress),
        )
        return

    uploaded_file = st.file_uploader(
        "Choose a ZIP or TAR archive",
        type=["zip", "tar", "gz", "tgz", "bz2", "xz"],
        key="files-archive-inspect-upload",
        help="Inspection lists archive entries and does not extract files.",
    )
    if uploaded_file is None:
        return
    archive_data = uploaded_file.getvalue()
    archive_name = uploaded_file.name
    st.caption("Archive contents will be provided as a CSV inventory.")
    _process_button(
        "Inspect archive",
        lambda progress: inspect_archive(archive_data, archive_name, progress),
    )


def _render_duplicate_finder() -> None:
    uploaded_files = _upload_files("duplicates")
    if not uploaded_files:
        return
    if len(uploaded_files) < 2:
        st.info("Choose at least two files to find duplicates.")
        return
    st.caption(
        "Files are compared by SHA-256 content, so matching files can have "
        "different names."
    )
    inputs = _file_inputs(uploaded_files)
    _process_button(
        "Find duplicates",
        lambda progress: find_duplicates(inputs, progress),
    )


def _format_size(size: int) -> str:
    size_value = float(size)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size_value < 1024 or unit == "TB":
            return f"{size_value:.1f} {unit}" if unit != "B" else f"{size} B"
        size_value /= 1024
    return f"{size} B"


def _render_size_analyzer() -> None:
    uploaded_files = _upload_files("size-analyzer")
    if not uploaded_files:
        return
    inputs = _file_inputs(uploaded_files)
    total_size = sum(len(data) for data, _ in inputs)
    st.metric("Files selected", len(inputs))
    st.metric("Combined size", _format_size(total_size))
    st.dataframe(
        [
            {"File": name, "Size": _format_size(len(data)), "Bytes": len(data)}
            for data, name in sorted(inputs, key=lambda item: len(item[0]), reverse=True)
        ],
        hide_index=True,
        use_container_width=True,
    )
    _process_button(
        "Create size report",
        lambda progress: analyze_file_sizes(inputs, progress),
    )


@st.fragment(run_every=0.25)
def render_file_tool_page() -> None:
    _poll_video_job()
    if st.button("← Back to Files", key="files-page-back"):
        _request_back_to_category()
        st.rerun()

    tool_name = st.session_state.get("active_tool")
    st.title(f"🗂️  {tool_name or 'File tool'}")
    st.divider()
    st.caption("File tools operate on uploaded files; no local folders are accessed.")
    _render_processing_progress()

    tools = {
        "File renamer": _render_renamer,
        "Archive manager": _render_archive_manager,
        "Duplicate finder": _render_duplicate_finder,
        "File size analyzer": _render_size_analyzer,
    }
    render_tool = tools.get(tool_name)
    if render_tool is None:
        st.error("This file tool is not available.")
        return
    render_tool()

    result = st.session_state.get("video_tool_result")
    if (
        isinstance(result, VideoResult)
        and st.session_state.get("video_processing_tool") == tool_name
    ):
        st.divider()
        _show_result(result)
