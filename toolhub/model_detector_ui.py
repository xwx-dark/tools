"""Streamlit page for running and browsing local YOLO detections."""

from pathlib import Path

import streamlit as st
from PIL import Image

from toolhub.annotator import render_annotated_image
from toolhub.detection_viewer import render_class_summary, render_zoomable_image
from toolhub.model_detector import (
    DetectionFrame,
    ModelDetectorResult,
    detect_with_yolo_model,
    read_video_frame,
)
from toolhub.model_tester import ModelEvaluationError
from toolhub.video import ProgressCallback, VideoResult
from toolhub.video_ui import (
    _poll_video_job,
    _process_button,
    _render_processing_progress,
    _request_back_to_category,
)
from .local_folder_picker import (
    LocalFolderPickerError,
    pick_local_file,
    pick_local_folder,
)


MODEL_DETECTOR_TOOL_NAMES = ("Model detector",)

_FRAME_KEY = "model-detector-frame"
_ZOOM_KEY = "model-detector-zoom"


def _clear_detector_result() -> None:
    st.session_state.pop("video_tool_result", None)
    st.session_state.pop("video_tool_error", None)
    st.session_state.pop("video_tool_error_tool", None)
    st.session_state["model-detector-results-open"] = False


def _move_frame(delta: int) -> None:
    result = st.session_state.get("video_tool_result")
    frames = result.frames if isinstance(result, ModelDetectorResult) else []
    current = int(st.session_state.get(_FRAME_KEY, 1))
    frame = min(len(frames), max(1, current + delta))
    st.session_state[_FRAME_KEY] = frame


def _change_zoom(delta: float) -> None:
    current = float(st.session_state.get(_ZOOM_KEY, 1.0))
    st.session_state[_ZOOM_KEY] = min(2.0, max(0.5, round(current + delta, 2)))


def _close_results_dialog() -> None:
    st.session_state["model-detector-results-open"] = False


def _result_frame_image(frame: DetectionFrame) -> Image.Image | str:
    if frame.frame_number is not None:
        return read_video_frame(frame.image_path, frame.frame_number)
    return frame.image_path


@st.dialog(
    "Model detector results",
    width="large",
    dismissible=True,
    on_dismiss=_close_results_dialog,
)
def _show_results_dialog(result: ModelDetectorResult) -> None:
    frames = result.frames
    if not frames:
        st.info("No frames are available in the detection results.")
        return

    current_frame = min(len(frames), max(1, int(st.session_state.get(_FRAME_KEY, 1))))
    st.session_state[_FRAME_KEY] = current_frame
    frame = frames[current_frame - 1]
    zoom = float(st.session_state.get(_ZOOM_KEY, 1.0))
    try:
        preview = render_annotated_image(
            _result_frame_image(frame),
            frame.detections,
            zoom,
            result.class_names,
            show_confidence=True,
        )
    except ModelEvaluationError as exc:
        st.error(str(exc))
        return

    image_panel, control_panel = st.columns([7, 3], gap="large")
    with image_panel:
        render_zoomable_image(preview, f"Detection preview {frame.image_key}")
        frame_description = (
            f"Frame {frame.frame_number + 1}"
            if frame.frame_number is not None
            else frame.image_key
        )
        st.caption(
            f"Result {current_frame} of {len(frames)} · {frame_description} · "
            f"{len(frame.detections)} detection(s)"
        )
        render_class_summary(
            frame.detections,
            result.class_names,
            "Detections by class",
        )

    with control_panel:
        with st.container(border=True):
            st.markdown("#### Result controls")
            previous, next_frame = st.columns(2)
            previous.button(
                " ",
                key="model-detector-previous",
                icon=":material/arrow_back:",
                help="Previous result (Left arrow)",
                shortcut="ArrowLeft",
                disabled=current_frame <= 1,
                on_click=_move_frame,
                args=(-1,),
                width="stretch",
            )
            next_frame.button(
                " ",
                key="model-detector-next",
                icon=":material/arrow_forward:",
                help="Next result (Right arrow)",
                shortcut="ArrowRight",
                disabled=current_frame >= len(frames),
                on_click=_move_frame,
                args=(1,),
                width="stretch",
            )
            jump_key = "model-detector-jump-frame"
            with st.popover(
                " ",
                icon=":material/image_search:",
                help="Go to a specific result",
                key="model-detector-go-to-frame",
                width="stretch",
            ):
                st.number_input(
                    "Result number",
                    min_value=1,
                    max_value=len(frames),
                    step=1,
                    key=jump_key,
                    format="%d",
                )
                if st.button(
                    " ",
                    key="model-detector-jump-submit",
                    icon=":material/check:",
                    help="Go to the selected result",
                    width="stretch",
                ):
                    st.session_state[_FRAME_KEY] = int(st.session_state[jump_key])

            st.divider()
            st.markdown("#### Zoom")
            zoom_out, zoom_in = st.columns(2)
            zoom_out.button(
                " ",
                key="model-detector-zoom-out",
                icon=":material/zoom_out:",
                help="Zoom out",
                disabled=zoom <= 0.5,
                on_click=_change_zoom,
                args=(-0.25,),
                width="stretch",
            )
            zoom_in.button(
                " ",
                key="model-detector-zoom-in",
                icon=":material/zoom_in:",
                help="Zoom in",
                disabled=zoom >= 2.0,
                on_click=_change_zoom,
                args=(0.25,),
                width="stretch",
            )
            st.caption(f"Zoom: {zoom:.0%}")
            st.caption("Use ← and → to navigate results.")
            if st.button(
                "Close results",
                key="model-detector-close-results",
                width="stretch",
            ):
                st.session_state["model-detector-results-open"] = False
                st.rerun(scope="app")


def _render_detector_inputs() -> tuple[str, str, str, float] | None:
    input_type_key = "model-detector-input-type"
    previous_input_type = st.session_state.get(input_type_key, "Image")
    input_type = st.selectbox(
        "Input selector",
        ("Image", "Video", "Path"),
        key=input_type_key,
        help="Path scans all supported images inside the selected folder and its subfolders.",
    )
    input_key = "model-detector-input-path"
    if input_key not in st.session_state:
        st.session_state[input_key] = ""
    if input_type != previous_input_type:
        st.session_state[input_key] = ""
        _clear_detector_result()

    browse_label = {
        "Image": "Choose image",
        "Video": "Choose video",
        "Path": "Choose image folder",
    }[input_type]
    if st.button(browse_label, key="model-detector-browse-input"):
        try:
            if input_type == "Path":
                selected = pick_local_folder(st.session_state[input_key] or None)
            else:
                filetypes = (
                    (("Images", "*.bmp *.jpeg *.jpg *.png *.tif *.tiff *.webp"),)
                    if input_type == "Image"
                    else (("Videos", "*.avi *.m4v *.mkv *.mov *.mp4 *.webm"),)
                )
                selected = pick_local_file(
                    Path(st.session_state[input_key]).parent
                    if st.session_state[input_key]
                    else None,
                    filetypes,
                    title=f"Choose {input_type.casefold()} input",
                )
        except LocalFolderPickerError as exc:
            st.error(str(exc))
        else:
            if selected:
                st.session_state[input_key] = selected
                _clear_detector_result()

    input_path = st.session_state[input_key]
    if input_path:
        st.code(input_path, language=None)
    else:
        st.info(f"Choose a local {input_type.casefold()} input to detect.")

    model_key = "model-detector-model-path"
    if model_key not in st.session_state:
        st.session_state[model_key] = ""
    if st.button("Choose YOLO model (.pt)", key="model-detector-browse-model"):
        try:
            selected_model = pick_local_file(
                Path(st.session_state[model_key]).parent
                if st.session_state[model_key]
                else None,
                (("PyTorch weights", "*.pt"),),
                title="Choose YOLO model weights",
            )
        except LocalFolderPickerError as exc:
            st.error(str(exc))
        else:
            if selected_model:
                st.session_state[model_key] = selected_model
                _clear_detector_result()
    model_path = st.session_state[model_key]
    if model_path:
        st.code(model_path, language=None)
    else:
        st.info("Select a trusted Ultralytics YOLO .pt model file.")

    confidence_threshold = st.slider(
        "Minimum confidence",
        min_value=0.01,
        max_value=0.99,
        value=0.25,
        step=0.01,
        key="model-detector-confidence",
        help="Detections below this confidence are omitted.",
    )
    trusted_model = st.checkbox(
        "I trust this model file and its source.",
        key="model-detector-trusted-model",
        help="Only load checkpoints you trust; serialized model files may contain code.",
    )
    if not trusted_model:
        st.warning("Confirm that the model file is trusted before running detection.")
    if not input_path or not model_path or not trusted_model:
        return None
    return input_type, input_path, model_path, confidence_threshold


@st.fragment(run_every=0.25)
def render_model_detector_page() -> None:
    _poll_video_job()
    if st.button("← Back to AI", key="model-detector-page-back"):
        _request_back_to_category()
        st.rerun()

    st.title("✨  Model detector")
    st.divider()
    st.caption(
        "Run a trusted Ultralytics YOLO model on one image, a video, or every "
        "supported image in a folder."
    )
    st.warning(
        "Model checkpoints may contain serialized code and detection can be "
        "CPU/GPU and memory intensive."
    )
    _render_processing_progress()
    selected = _render_detector_inputs()
    if selected is not None:
        input_type, input_path, model_path, confidence_threshold = selected

        def detect(progress: ProgressCallback) -> VideoResult:
            return detect_with_yolo_model(
                input_type,
                input_path,
                model_path,
                confidence_threshold,
                progress,
            )

        _process_button("Detect objects", detect)

    result = st.session_state.get("video_tool_result")
    if (
        isinstance(result, ModelDetectorResult)
        and st.session_state.get("video_processing_tool") == "Model detector"
    ):
        detections = sum(len(frame.detections) for frame in result.frames)
        st.divider()
        st.success(
            f"Detected {detections} object(s) in {len(result.frames)} "
            f"{'frame(s)' if result.input_type == 'Video' else 'image(s)'}."
        )
        st.caption(f"Model: {result.model_name} · Input: {result.input_path}")
        st.download_button(
            "Download detection summary",
            data=result.data,
            file_name=result.filename,
            mime=result.mime_type,
        )
        if st.button(
            "Show results",
            key="model-detector-show-results",
            type="primary",
        ):
            st.session_state[_FRAME_KEY] = 1
            st.session_state[_ZOOM_KEY] = 1.0
            st.session_state["model-detector-results-open"] = True
        if st.session_state.get("model-detector-results-open"):
            _show_results_dialog(result)
