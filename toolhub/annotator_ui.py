"""Streamlit page for browsing images with their ground-truth annotations."""

import streamlit as st

from toolhub.annotator import load_annotator_dataset, render_annotated_image
from .local_folder_picker import LocalFolderPickerError, pick_local_folder
from toolhub.model_tester import ModelEvaluationError
from toolhub.video_ui import _request_back_to_category


ANNOTATOR_TOOL_NAMES = ("Label applier",)

_FRAME_KEY = "annotator-frame-number"
_JUMP_FRAME_KEY = "annotator-jump-frame-number"
_ZOOM_KEY = "annotator-zoom"


def _move_frame(delta: int) -> None:
    samples = st.session_state.get("annotator-samples", [])
    current = int(st.session_state.get(_FRAME_KEY, 1))
    frame = min(len(samples), max(1, current + delta))
    st.session_state[_FRAME_KEY] = frame
    st.session_state[_JUMP_FRAME_KEY] = frame


def _change_zoom(delta: float) -> None:
    current = float(st.session_state.get(_ZOOM_KEY, 1.0))
    st.session_state[_ZOOM_KEY] = min(2.0, max(0.5, round(current + delta, 2)))


def _close_results_dialog() -> None:
    st.session_state["annotator-results-open"] = False


def _clear_loaded_dataset() -> None:
    st.session_state.pop("annotator-samples", None)
    st.session_state.pop("annotator-annotations", None)
    st.session_state.pop("annotator-class-names", None)
    st.session_state.pop("annotator-loaded-path", None)
    st.session_state["annotator-results-open"] = False


def _load_folder(dataset_path: str) -> None:
    _clear_loaded_dataset()
    try:
        samples, annotations, class_names = load_annotator_dataset(dataset_path)
    except ModelEvaluationError as exc:
        st.session_state["annotator-load-error"] = str(exc)
        return
    st.session_state["annotator-samples"] = [
        (str(image_path), image_key) for image_path, image_key in samples
    ]
    st.session_state["annotator-annotations"] = annotations
    st.session_state["annotator-class-names"] = class_names
    st.session_state["annotator-loaded-path"] = dataset_path
    st.session_state[_FRAME_KEY] = 1
    st.session_state[_JUMP_FRAME_KEY] = 1
    st.session_state[_ZOOM_KEY] = 1.0
    st.session_state.pop("annotator-load-error", None)


@st.dialog("Label applier results", width="large")
def _show_results_dialog() -> None:
    samples = st.session_state.get("annotator-samples", [])
    annotations_by_image = st.session_state.get("annotator-annotations", {})
    class_names = st.session_state.get("annotator-class-names", {})
    if not samples:
        st.info("No images are available in this folder.")
        return

    current_frame = min(
        len(samples), max(1, int(st.session_state.get(_FRAME_KEY, 1)))
    )
    st.session_state[_FRAME_KEY] = current_frame
    zoom = float(st.session_state.get(_ZOOM_KEY, 1.0))
    image_path, image_key = samples[current_frame - 1]
    boxes = annotations_by_image.get(image_key, [])
    try:
        preview = render_annotated_image(image_path, boxes, zoom, class_names)
    except ModelEvaluationError as exc:
        st.error(str(exc))
        return

    image_panel, control_panel = st.columns([7, 3], gap="large")
    with image_panel:
        st.image(preview, width="content")
        st.caption(
            f"Image {current_frame} of {len(samples)} · {image_key} · "
            f"{len(boxes)} annotation(s)"
        )
        st.caption(
            "Labels show class IDs from the annotation files; no class-name "
            "mapping is included in label rows. Names are loaded from "
            "data.yaml/data.yml or classes.txt when available."
        )

    with control_panel:
        with st.container(border=True):
            st.markdown("#### Image controls")
            previous, next_image = st.columns(2)
            previous.button(
                " ",
                key="annotator-previous",
                icon=":material/arrow_back:",
                help="Previous image (Left arrow)",
                shortcut="ArrowLeft",
                disabled=current_frame <= 1,
                on_click=_move_frame,
                args=(-1,),
                width="stretch",
            )
            next_image.button(
                " ",
                key="annotator-next",
                icon=":material/arrow_forward:",
                help="Next image (Right arrow)",
                shortcut="ArrowRight",
                disabled=current_frame >= len(samples),
                on_click=_move_frame,
                args=(1,),
                width="stretch",
            )

            with st.popover(
                " ",
                icon=":material/image_search:",
                help="Go to a specific image",
                key="annotator-go-to-image",
                width="stretch",
            ):
                st.number_input(
                    "Image number",
                    min_value=1,
                    max_value=len(samples),
                    step=1,
                    key=_JUMP_FRAME_KEY,
                    format="%d",
                    help=f"Choose an image number from 1 to {len(samples)}.",
                )
                if st.button(
                    " ",
                    key="annotator-jump-submit",
                    icon=":material/check:",
                    help="Go to the selected image",
                    width="stretch",
                ):
                    st.session_state[_FRAME_KEY] = int(
                        st.session_state[_JUMP_FRAME_KEY]
                    )

            st.divider()
            st.markdown("#### Zoom")
            zoom_out, zoom_in = st.columns(2)
            zoom_out.button(
                " ",
                key="annotator-zoom-out",
                icon=":material/zoom_out:",
                help="Zoom out",
                disabled=zoom <= 0.5,
                on_click=_change_zoom,
                args=(-0.25,),
                width="stretch",
            )
            zoom_in.button(
                " ",
                key="annotator-zoom-in",
                icon=":material/zoom_in:",
                help="Zoom in",
                disabled=zoom >= 2.0,
                on_click=_change_zoom,
                args=(0.25,),
                width="stretch",
            )
            st.caption(f"Zoom: {zoom:.0%}")
            st.caption("Use ← and → to navigate between images.")
            st.divider()
            if st.button(
                " ",
                key="annotator-close-results",
                icon=":material/close:",
                help="Close results",
                width="stretch",
            ):
                _close_results_dialog()
                st.rerun(scope="app")


def render_annotator_page() -> None:
    if st.button("← Back to AI", key="annotator-page-back"):
        _clear_loaded_dataset()
        _request_back_to_category()
        st.rerun()

    st.title("✨  Label applier")
    st.divider()
    st.caption(
        "Browse labeled images with their YOLO bounding boxes drawn over the image."
    )
    if "annotator-dataset-path" not in st.session_state:
        st.session_state["annotator-dataset-path"] = ""
    if st.button("Browse for dataset folder", key="annotator-browse-folder"):
        try:
            selected_path = pick_local_folder(
                st.session_state["annotator-dataset-path"] or None
            )
        except LocalFolderPickerError as exc:
            st.session_state["annotator-load-error"] = str(exc)
        else:
            if selected_path:
                if selected_path != st.session_state["annotator-dataset-path"]:
                    _clear_loaded_dataset()
                st.session_state["annotator-dataset-path"] = selected_path
                st.session_state.pop("annotator-load-error", None)

    dataset_path = st.session_state["annotator-dataset-path"]
    if dataset_path:
        st.code(dataset_path, language=None)
    else:
        st.info("Browse to choose a local folder containing images/ and labels/.")

    if st.button("Load folder", key="annotator-load-folder", type="primary"):
        if not dataset_path:
            st.session_state["annotator-load-error"] = (
                "Browse to choose a folder containing images/ and labels/ first."
            )
        else:
            with st.spinner("Validating images and annotations..."):
                _load_folder(dataset_path)

    error = st.session_state.get("annotator-load-error")
    if error:
        st.error(error)

    samples = st.session_state.get("annotator-samples", [])
    if not samples:
        st.info("No images are available in this folder.")
        return

    st.success(f"Loaded {len(samples)} image(s) and their annotations.")
    if st.button("Show results", key="annotator-show-results", type="primary"):
        st.session_state["annotator-results-open"] = True
    if st.session_state.get("annotator-results-open"):
        _show_results_dialog()
