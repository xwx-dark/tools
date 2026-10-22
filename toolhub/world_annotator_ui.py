"""Streamlit page for generating YOLO annotations."""

import json
from concurrent.futures import Future
from pathlib import Path

import streamlit as st

from toolhub.model_tester import ModelEvaluationError
from toolhub.video import ProgressCallback, VideoResult
from toolhub.video_ui import (
    _poll_video_job,
    _process_button,
    _render_processing_progress,
    _request_back_to_category,
)
from toolhub.world_annotator import (
    annotate_dataset_with_yolo_world,
    existing_output_files,
    find_dataset_images,
    list_ollama_vision_models,
)
from .local_folder_picker import (
    LocalFolderPickerError,
    pick_local_file,
    pick_local_folder,
)


WORLD_ANNOTATOR_TOOL_NAMES = ("Annotator",)


def _parse_class_names(value: str) -> list[str]:
    return [name.strip() for name in value.splitlines() if name.strip()]


def _show_annotation_report(result: VideoResult) -> None:
    try:
        report = json.loads(result.data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        st.error(f"Could not display the annotation report: {exc}")
        return

    st.success(
        f"Annotated {report['images']} image(s), producing "
        f"{report['detections']} detection(s)."
    )
    if report.get("ignored_detections", 0):
        st.warning(
            f"Ignored {report['ignored_detections']} invalid detection(s) returned "
            "by the model. Review the generated labels before training."
        )
    st.code(report["labels_directory"], language=None)
    st.caption(
        f"Class names saved in {report['class_names_file']} · "
        f"{report['overwritten_files']} existing output file(s) replaced."
    )
    st.caption(f"Backend: {report['backend']} · Model: {report['model']}")
    st.dataframe(
        [
            {"Class ID": class_id, "Class": name}
            for class_id, name in enumerate(report["classes"])
        ],
        hide_index=True,
        use_container_width=True,
    )
    st.download_button(
        "Download annotation report",
        data=result.data,
        file_name=result.filename,
        mime=result.mime_type,
    )


@st.fragment(run_every=0.25)
def render_world_annotator_page() -> None:
    _poll_video_job()
    if st.button("← Back to AI", key="world-annotator-page-back"):
        _request_back_to_category()
        st.rerun()

    st.title("✨  Annotator")
    st.divider()
    ollama_model: str | None = None
    backend = st.selectbox(
        "Annotation backend",
        ("YOLO-World", "Ollama"),
        key="world-annotator-backend",
    )
    if backend == "YOLO-World":
        st.caption(
            "Generate YOLO-format image labels from custom text classes using "
            "Ultralytics YOLO-World."
        )
        st.info(
            "Select a dataset folder containing images/. Labels are saved under "
            "labels/ using the same relative image paths. First-time setup requires "
            "the CLIP package and detector/text-encoder weights; see the setup help "
            "below if this machine has restricted network access."
        )
    else:
        st.caption(
            "Use an installed Ollama vision model to generate YOLO-format labels."
        )
        st.info(
            "Ollama models estimate bounding boxes from image descriptions, so "
            "their coordinates and confidence values can be unreliable and may "
            "repeat similar boxes across images. This is a limitation of vision "
            "language models, not a label-coordinate conversion: use YOLO-World "
            "for object-detection annotations and review generated labels before "
            "training."
        )

        ollama_models_key = "world-annotator-ollama-vision-models"
        ollama_error_key = "world-annotator-ollama-model-error"
        refresh_models = st.button(
            "Refresh local Ollama models",
            key="world-annotator-refresh-ollama-models",
        )
        if refresh_models or ollama_models_key not in st.session_state:
            try:
                st.session_state[ollama_models_key] = list_ollama_vision_models()
                st.session_state.pop(ollama_error_key, None)
            except ModelEvaluationError as exc:
                st.session_state[ollama_models_key] = []
                st.session_state[ollama_error_key] = str(exc)
        if st.session_state.get(ollama_error_key):
            st.error(st.session_state[ollama_error_key])
        ollama_models = st.session_state.get(ollama_models_key, [])
        ollama_model = None
        if ollama_models:
            ollama_model = st.selectbox(
                "Installed Ollama vision model",
                ollama_models,
                key="world-annotator-ollama-model",
            )
        else:
            st.warning(
                "No installed vision-capable Ollama models were found. Start "
                "Ollama, install a vision model, then refresh the model list."
            )

    folder_key = "world-annotator-dataset-path"
    if folder_key not in st.session_state:
        st.session_state[folder_key] = ""
    if st.button("Browse for dataset folder", key="world-annotator-browse-folder"):
        try:
            selected_path = pick_local_folder(
                st.session_state[folder_key] or None
            )
        except LocalFolderPickerError as exc:
            st.error(str(exc))
        else:
            if selected_path:
                st.session_state[folder_key] = selected_path

    dataset_path = st.session_state[folder_key]
    if dataset_path:
        st.code(dataset_path, language=None)
    else:
        st.info("Browse to choose a folder containing an images/ directory.")

    weights_path = None
    if backend == "YOLO-World":
        weights_key = "world-annotator-weights-path"
        if weights_key not in st.session_state:
            st.session_state[weights_key] = ""
        if st.button(
            "Choose local YOLO-World weights (optional)",
            key="world-annotator-browse-weights",
        ):
            try:
                selected_weights = pick_local_file(
                    Path(st.session_state[weights_key]).parent
                    if st.session_state[weights_key]
                    else None,
                    (("PyTorch weights", "*.pt"), ("All files", "*.*")),
                )
            except LocalFolderPickerError as exc:
                st.error(str(exc))
            else:
                if selected_weights:
                    st.session_state[weights_key] = selected_weights
        weights_path = st.session_state[weights_key] or None
        if weights_path:
            st.code(weights_path, language=None)
            if st.button(
                "Clear local weights selection",
                key="world-annotator-clear-weights",
            ):
                st.session_state[weights_key] = ""
                st.rerun()
        with st.expander("Network-restricted or offline setup"):
            st.markdown(
                "Install the project dependencies while connected to a package source "
                "that provides the Ultralytics CLIP dependency. The first run also "
                "downloads YOLO-World detector weights and CLIP ViT-B/32 text weights. "
                "You may select a local YOLO-World `.pt` file above to avoid the "
                "detector download. Fully offline operation additionally requires "
                "CLIP's `ViT-B-32.pt` cache file in Ultralytics' configured "
                "`weights_dir/clip` directory. A local detector file alone does not "
                "replace the CLIP package or its text weights."
            )

    class_text = st.text_area(
        "Custom classes",
        key="world-annotator-classes",
        placeholder="person\ncar\nbicycle",
        help="Enter one class name per line. These lines determine YOLO class IDs.",
    )
    class_names = _parse_class_names(class_text)
    normalized_names = [name.casefold() for name in class_names]
    classes_valid = bool(class_names) and len(set(normalized_names)) == len(class_names)
    if not class_names:
        st.caption("Add at least one class, with one class name per line.")
    elif not classes_valid:
        st.error("Class names must be unique, ignoring letter case.")

    confidence = st.slider(
        "Minimum confidence",
        min_value=0.01,
        max_value=0.99,
        value=0.25,
        step=0.01,
        key="world-annotator-confidence",
        help="Detections below this confidence are excluded from saved labels.",
    )

    image_count = 0
    existing_files = []
    dataset_valid = False
    if dataset_path:
        try:
            image_count = len(find_dataset_images(dataset_path))
            existing_files = existing_output_files(dataset_path)
            dataset_valid = True
            st.caption(f"{image_count} supported image(s) found.")
        except ModelEvaluationError as exc:
            st.error(str(exc))

    overwrite_confirmed = False
    if existing_files:
        st.warning(
            f"{len(existing_files)} existing output file(s) will be replaced, "
            "including matching labels and/or the class-name mapping."
        )
        overwrite_confirmed = st.checkbox(
            "I understand and confirm replacing these existing output files.",
            key="world-annotator-confirm-overwrite",
        )

    can_process = dataset_valid and classes_valid and (
        not existing_files or overwrite_confirmed
    ) and (backend != "Ollama" or ollama_model is not None)
    captured_path = dataset_path
    captured_classes = class_names.copy()
    captured_backend = backend
    captured_ollama_model = ollama_model

    def run_annotation(progress: ProgressCallback) -> VideoResult:
        return annotate_dataset_with_yolo_world(
            captured_path,
            captured_classes,
            confidence,
            overwrite_confirmed,
            progress,
            weights_path,
            backend=captured_backend,
            ollama_model=captured_ollama_model,
        )

    _render_processing_progress()
    _process_button(
        "Annotate images",
        run_annotation,
        disabled=not can_process,
    )

    future = st.session_state.get("video_processing_future")
    busy = isinstance(future, Future)
    result = st.session_state.get("video_tool_result")
    if (
        not busy
        and isinstance(result, VideoResult)
        and st.session_state.get("video_processing_tool") == "Annotator"
    ):
        st.divider()
        _show_annotation_report(result)
