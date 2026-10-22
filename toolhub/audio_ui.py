"""Page-based user interfaces for the audio tools."""

from pathlib import Path
import tempfile

import streamlit as st

from toolhub.audio import (
    _probe_audio,
    convert_audio,
    merge_audio,
    normalize_audio,
    trim_audio,
)
from toolhub.video import VideoProcessingError
from toolhub.video_ui import (
    _process_button,
    _render_processing_progress,
    _request_back_to_category,
    _show_result,
    _poll_video_job,
)


AUDIO_TOOL_NAMES = (
    "Audio converter",
    "Audio trimmer",
    "Volume normalizer",
    "Audio merger",
)

_AUDIO_FILE_TYPES = ["mp3", "wav", "aac", "m4a", "flac", "ogg", "opus"]


def _upload_audio(tool_key: str, *, multiple: bool = False):
    return st.file_uploader(
        "Choose audio files" if multiple else "Choose an audio file",
        type=_AUDIO_FILE_TYPES,
        accept_multiple_files=multiple,
        key=f"audio-{tool_key}-upload",
        help="FFmpeg processes the files on the machine running this app.",
    )


def _get_audio_duration(uploaded_file) -> float | None:
    if uploaded_file is None:
        return None
    try:
        with tempfile.TemporaryDirectory(prefix="everyday-tools-audio-probe-") as directory:
            path = Path(directory) / (uploaded_file.name or "audio.mp3")
            path.write_bytes(uploaded_file.getvalue())
            duration = _probe_audio(path)
    except VideoProcessingError as exc:
        st.error(str(exc))
        return None
    st.caption(f"Audio duration: {duration:.2f} seconds")
    return duration


def _render_converter() -> None:
    uploaded_file = _upload_audio("converter")
    output_format = st.selectbox(
        "Convert to",
        options=["MP3", "WAV", "AAC", "FLAC", "OGG", "M4A"],
        key="audio-converter-format",
    )
    if uploaded_file is None:
        return
    input_data = uploaded_file.getvalue()
    input_name = uploaded_file.name
    _process_button(
        "Convert audio",
        lambda progress: convert_audio(
            input_data, input_name, output_format.lower(), progress
        ),
    )


def _render_trimmer() -> None:
    uploaded_file = _upload_audio("trimmer")
    duration = _get_audio_duration(uploaded_file)
    if uploaded_file is None or duration is None:
        return
    if duration < 0.1:
        st.info("This audio is too short to trim.")
        return
    start_seconds = st.number_input(
        "Start time (seconds)",
        min_value=0.0,
        max_value=float(duration),
        value=0.0,
        step=0.1,
        key="audio-trimmer-start",
    )
    end_seconds = st.number_input(
        "End time (seconds)",
        min_value=0.1,
        max_value=float(duration),
        value=float(duration),
        step=0.1,
        key="audio-trimmer-end",
    )
    input_data = uploaded_file.getvalue()
    input_name = uploaded_file.name
    _process_button(
        "Trim audio",
        lambda progress: trim_audio(
            input_data, input_name, start_seconds, end_seconds, progress
        ),
    )


def _render_normalizer() -> None:
    uploaded_file = _upload_audio("normalizer")
    target_lufs = st.slider(
        "Target loudness (LUFS)",
        min_value=-24,
        max_value=-14,
        value=-16,
        help="A common target for general audio is around -16 LUFS.",
        key="audio-normalizer-lufs",
    )
    if uploaded_file is None:
        return
    input_data = uploaded_file.getvalue()
    input_name = uploaded_file.name
    _process_button(
        "Normalize volume",
        lambda progress: normalize_audio(
            input_data, input_name, target_lufs, progress
        ),
    )


def _render_merger() -> None:
    uploaded_files = _upload_audio("merger", multiple=True)
    output_format = st.selectbox(
        "Merged output format",
        options=["MP3", "WAV", "M4A"],
        key="audio-merger-format",
    )
    if uploaded_files and len(uploaded_files) < 2:
        st.info("Choose at least two audio files to merge.")
        return
    if not uploaded_files:
        return
    inputs = [(file.getvalue(), file.name) for file in uploaded_files]
    _process_button(
        "Merge audio",
        lambda progress: merge_audio(inputs, output_format.lower(), progress),
    )


@st.fragment(run_every=0.25)
def render_audio_tool_page() -> None:
    _poll_video_job()
    if st.button("← Back to Audio", key="audio-page-back"):
        _request_back_to_category()
        st.rerun()

    tool_name = st.session_state.get("active_tool")
    st.title(f"🎧  {tool_name or 'Audio tool'}")
    st.divider()
    st.caption("FFmpeg processes the file on the machine running this app.")
    _render_processing_progress()

    tools = {
        "Audio converter": _render_converter,
        "Audio trimmer": _render_trimmer,
        "Volume normalizer": _render_normalizer,
        "Audio merger": _render_merger,
    }
    render_tool = tools.get(tool_name)
    if render_tool is None:
        st.error("This audio tool is not available.")
        return
    render_tool()

    result = st.session_state.get("video_tool_result")
    if (
        st.session_state.get("video_processing_tool") == tool_name
        and result is not None
    ):
        st.divider()
        _show_result(result)
