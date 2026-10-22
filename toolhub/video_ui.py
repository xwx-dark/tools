"""Page-based user interfaces and shared processing state for media tools."""

from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from queue import Empty, Queue
import tempfile
from typing import Callable

import streamlit as st

from toolhub.video import (
    VideoInfo,
    VideoProcessingError,
    VideoResult,
    compress_video,
    convert_video,
    extract_thumbnail,
    make_gif,
    probe_video,
    trim_video,
)


VIDEO_TOOL_NAMES = (
    "Video compressor",
    "Video converter",
    "Video trimmer",
    "GIF maker",
    "Thumbnail extractor",
)

_VIDEO_EXECUTOR = ThreadPoolExecutor(max_workers=2, thread_name_prefix="video-tool")


def _open_tool(tool_name: str, category: str) -> None:
    st.session_state["active_tool"] = tool_name
    st.session_state["active_tool_category"] = category
    st.session_state["selected_category"] = category
    st.session_state.pop("pending_video_navigation", None)
    st.session_state.pop("video_tool_result", None)
    st.session_state.pop("video_tool_error", None)


def _video_job_is_running() -> bool:
    _poll_video_job()
    future = st.session_state.get("video_processing_future")
    return isinstance(future, Future) and not future.done()


def handle_category_change() -> None:
    """Keep the active tool open until unfinished processing is confirmed."""
    if not st.session_state.get("active_tool"):
        return

    destination = st.session_state.get("selected_category", "Video")
    if _video_job_is_running():
        st.session_state["pending_video_navigation"] = destination
        st.session_state["selected_category"] = st.session_state.get(
            "active_tool_category", "Video"
        )
    else:
        st.session_state.pop("active_tool", None)
        st.session_state.pop("pending_video_navigation", None)


def _request_back_to_category() -> None:
    if _video_job_is_running():
        st.session_state["pending_video_navigation"] = st.session_state.get(
            "active_tool_category", "Video"
        )
    else:
        st.session_state.pop("active_tool", None)
        st.session_state.pop("pending_video_navigation", None)


def _confirm_video_navigation() -> None:
    destination = st.session_state.get("pending_video_navigation", "Video")
    st.session_state["selected_category"] = destination
    st.session_state.pop("active_tool", None)
    st.session_state.pop("pending_video_navigation", None)


def _cancel_video_navigation() -> None:
    st.session_state["selected_category"] = st.session_state.get(
        "active_tool_category", "Video"
    )
    st.session_state.pop("pending_video_navigation", None)


@st.dialog("Leave this tool?")
def render_video_navigation_warning() -> None:
    st.warning(
        "A process is still running. If you leave this page, its result "
        "might not be saved."
    )
    st.write("Do you want to continue navigating away from this tool?")
    leave_col, stay_col = st.columns(2)
    with leave_col:
        st.button(
            "Continue navigating",
            type="primary",
            use_container_width=True,
            on_click=_confirm_video_navigation,
        )
    with stay_col:
        st.button(
            "Stay on this tool",
            use_container_width=True,
            on_click=_cancel_video_navigation,
        )


def _upload_video(tool_name: str):
    uploaded_file = st.file_uploader(
        "Choose a video",
        type=["mp4", "mov", "mkv", "avi", "webm", "m4v"],
        key=f"{tool_name}-upload",
        help="FFmpeg processes the file on the machine running this app.",
    )
    return uploaded_file


def _get_video_info(uploaded_file) -> VideoInfo | None:
    if uploaded_file is None:
        return None
    try:
        with st.spinner("Reading video details..."):
            with tempfile.TemporaryDirectory(prefix="everyday-tools-probe-") as directory:
                path = Path(directory) / (uploaded_file.name or "video.mp4")
                path.write_bytes(uploaded_file.getvalue())
                return probe_video(path)
    except VideoProcessingError as exc:
        st.error(str(exc))
        return None


def _show_result(result: VideoResult) -> None:
    st.success("Your file is ready.")
    st.download_button(
        "Download file",
        data=result.data,
        file_name=result.filename,
        mime=result.mime_type,
        type="primary",
        use_container_width=True,
    )


def _process_button(
    label: str,
    operation: Callable[[Callable[[float], None]], VideoResult],
    disabled: bool = False,
) -> None:
    _poll_video_job()
    future = st.session_state.get("video_processing_future")
    busy = isinstance(future, Future)

    error = st.session_state.get("video_tool_error")
    if error and st.session_state.get("video_tool_error_tool") == st.session_state.get(
        "active_tool"
    ):
        st.error(error)

    if st.button(
        label,
        type="primary",
        use_container_width=True,
        disabled=busy or disabled,
    ) and not busy:
        st.session_state.pop("video_tool_result", None)
        st.session_state.pop("video_tool_error", None)
        progress_queue: Queue[float] = Queue()
        st.session_state["video_processing_progress_queue"] = progress_queue
        st.session_state["video_processing_progress"] = 0.0
        st.session_state["video_processing_tool"] = st.session_state.get("active_tool")
        st.session_state["video_processing_future"] = _VIDEO_EXECUTOR.submit(
            operation, progress_queue.put
        )
        st.rerun()


def _render_processing_progress() -> None:
    future = st.session_state.get("video_processing_future")
    if not isinstance(future, Future):
        return

    progress = st.session_state.get("video_processing_progress", 0.0)
    visible_progress = max(0.02, min(1.0, progress))
    if progress <= 0:
        status = "Starting processing..."
    else:
        status = f"Processing... {progress:.0%}"
    st.progress(visible_progress, text=status)


def _poll_video_job() -> None:
    progress_queue = st.session_state.get("video_processing_progress_queue")
    if isinstance(progress_queue, Queue):
        while True:
            try:
                st.session_state["video_processing_progress"] = min(
                    1.0, max(0.0, progress_queue.get_nowait())
                )
            except Empty:
                break

    future = st.session_state.get("video_processing_future")
    if not isinstance(future, Future) or not future.done():
        return

    processing_tool = st.session_state.get("video_processing_tool")
    try:
        result = future.result()
    except (VideoProcessingError, ValueError) as exc:
        st.session_state["video_tool_error"] = str(exc)
        st.session_state["video_tool_error_tool"] = processing_tool
    except Exception as exc:
        st.session_state["video_tool_error"] = (
            f"Unexpected processing error: {exc}"
        )
        st.session_state["video_tool_error_tool"] = processing_tool
    else:
        st.session_state["video_tool_result"] = result
        st.session_state["video_processing_progress"] = 1.0
    finally:
        st.session_state.pop("video_processing_future", None)
        st.session_state.pop("video_processing_progress_queue", None)


def _render_compressor() -> None:
    st.write("Choose how much smaller you want the output to be.")
    uploaded_file = _upload_video("compressor")
    reduction_percent = st.slider(
        "Target size reduction",
        min_value=1,
        max_value=90,
        value=40,
        format="%d%%",
        key="compressor-reduction",
        help="The output size is an estimate; the final size depends on the video.",
    )
    if uploaded_file is not None:
        input_data = uploaded_file.getvalue()
        input_name = uploaded_file.name
        _process_button(
            "Compress video",
            lambda progress_callback: compress_video(
                input_data, input_name, reduction_percent, progress_callback
            ),
        )


def _render_converter() -> None:
    uploaded_file = _upload_video("converter")
    output_format = st.selectbox(
        "Convert to",
        options=["MP4", "MKV", "MOV", "WebM"],
        key="converter-format",
    )
    if uploaded_file is not None:
        input_data = uploaded_file.getvalue()
        input_name = uploaded_file.name
        _process_button(
            "Convert video",
            lambda progress_callback: convert_video(
                input_data, input_name, output_format, progress_callback
            ),
        )


def _render_trimmer() -> None:
    uploaded_file = _upload_video("trimmer")
    info = _get_video_info(uploaded_file)
    if info is None:
        return
    if info.duration < 0.1:
        st.info("This video is too short to trim.")
        return
    st.caption(f"Video duration: {info.duration:.2f} seconds")
    start_seconds = st.number_input(
        "Start time (seconds)",
        min_value=0.0,
        max_value=float(info.duration),
        value=0.0,
        step=0.1,
        key="trimmer-start",
    )
    end_seconds = st.number_input(
        "End time (seconds)",
        min_value=0.1,
        max_value=float(info.duration),
        value=float(info.duration),
        step=0.1,
        key="trimmer-end",
    )
    input_data = uploaded_file.getvalue()
    input_name = uploaded_file.name
    _process_button(
        "Trim video",
        lambda progress_callback: trim_video(
            input_data,
            input_name,
            start_seconds,
            end_seconds,
            progress_callback,
        ),
    )


def _render_gif_maker() -> None:
    uploaded_file = _upload_video("gif")
    info = _get_video_info(uploaded_file)
    if info is None:
        return
    if info.duration < 1:
        st.info("GIF creation requires a video that is at least 1 second long.")
        return
    start_maximum = max(0.0, info.duration - 1.0)
    start_seconds = st.number_input(
        "Start time (seconds)",
        min_value=0.0,
        max_value=float(start_maximum),
        value=0.0,
        step=0.1,
        key="gif-start",
    )
    remaining_duration = max(1.0, info.duration - start_seconds)
    duration_seconds = st.slider(
        "GIF duration (seconds)",
        min_value=1,
        max_value=min(30, int(remaining_duration)),
        value=min(5, max(1, int(remaining_duration))),
        key="gif-duration",
    )
    fps = st.slider("Frame rate", min_value=5, max_value=20, value=12, key="gif-fps")
    width = st.slider(
        "Output width (pixels)",
        min_value=160,
        max_value=960,
        value=480,
        step=40,
        key="gif-width",
    )
    input_data = uploaded_file.getvalue()
    input_name = uploaded_file.name
    _process_button(
        "Create GIF",
        lambda progress_callback: make_gif(
            input_data,
            input_name,
            start_seconds,
            float(duration_seconds),
            fps,
            width,
            progress_callback,
        ),
    )


def _render_thumbnail_extractor() -> None:
    uploaded_file = _upload_video("thumbnail")
    info = _get_video_info(uploaded_file)
    if info is None:
        return
    if info.duration < 0.05:
        st.info("This video is too short to extract a thumbnail.")
        return
    timestamp_seconds = st.number_input(
        "Frame time (seconds)",
        min_value=0.0,
        max_value=max(0.0, float(info.duration) - 0.05),
        value=min(1.0, max(0.0, float(info.duration) - 0.05)),
        step=0.1,
        key="thumbnail-time",
    )
    image_format = st.selectbox(
        "Image format", options=["JPG", "PNG"], key="thumbnail-format"
    )
    quality = st.slider(
        "JPG quality",
        min_value=1,
        max_value=100,
        value=90,
        key="thumbnail-quality",
        disabled=image_format == "PNG",
    )
    input_data = uploaded_file.getvalue()
    input_name = uploaded_file.name
    _process_button(
        "Extract thumbnail",
        lambda progress_callback: extract_thumbnail(
            input_data,
            input_name,
            timestamp_seconds,
            image_format.lower(),
            quality,
            progress_callback,
        ),
    )


@st.fragment(run_every=0.25)
def render_video_tool_page() -> None:
    _poll_video_job()
    category = st.session_state.get("active_tool_category", "Video")
    if st.button(f"← Back to {category}", key="video-page-back"):
        _request_back_to_category()
        st.rerun()

    tool_name = st.session_state.get("active_tool")
    st.title(f"🎬  {tool_name or 'Video tool'}")
    st.divider()
    st.caption("FFmpeg processes the file on the machine running this app.")
    _render_processing_progress()

    tools = {
        "Video compressor": _render_compressor,
        "Video converter": _render_converter,
        "Video trimmer": _render_trimmer,
        "GIF maker": _render_gif_maker,
        "Thumbnail extractor": _render_thumbnail_extractor,
    }
    render_tool = tools.get(tool_name)
    if render_tool is None:
        st.error("This video tool is not available.")
        return

    render_tool()
    result = st.session_state.get("video_tool_result")
    result_tool = st.session_state.get("video_processing_tool")
    active_tool = st.session_state.get("active_tool")
    if isinstance(result, VideoResult) and result_tool == active_tool:
        st.divider()
        _show_result(result)
