"""Streamlit UI for YOLO model evaluation and analytics."""

import json
from typing import Any

import streamlit as st

from .local_folder_picker import LocalFolderPickerError, pick_local_folder
from toolhub.model_tester import evaluate_yolo_model
from toolhub.video import ProgressCallback, VideoResult
from toolhub.video_ui import (
    _poll_video_job,
    _process_button,
    _render_processing_progress,
    _request_back_to_category,
    _show_result,
)


MODEL_TESTER_TOOL_NAMES = ("Model tester",)


def _evaluate_dataset(
    dataset_path: str,
    model_path: str,
    confidence_threshold: float,
    progress: ProgressCallback,
) -> VideoResult:
    return evaluate_yolo_model(
        dataset_path,
        model_path,
        confidence_threshold,
        progress,
    )


def _render_model_tester() -> None:
    if "model-tester-dataset-path" not in st.session_state:
        st.session_state["model-tester-dataset-path"] = ""
    if st.button("Browse for dataset folder", key="model-tester-browse-folder"):
        try:
            selected_path = pick_local_folder(
                st.session_state["model-tester-dataset-path"] or None
            )
        except LocalFolderPickerError as exc:
            st.error(str(exc))
        else:
            if selected_path and selected_path != st.session_state[
                "model-tester-dataset-path"
            ]:
                st.session_state["model-tester-dataset-path"] = selected_path
                st.session_state.pop("video_tool_result", None)
                if st.session_state.get("video_tool_error_tool") == "Model tester":
                    st.session_state.pop("video_tool_error", None)
                    st.session_state.pop("video_tool_error_tool", None)

    dataset_path = st.session_state["model-tester-dataset-path"]
    if dataset_path:
        st.code(dataset_path, language=None)
    else:
        st.info("Browse to choose a local folder containing images/ and labels/.")

    model_path = st.text_input(
        "Ultralytics YOLO model path (.pt)",
        placeholder=r"C:\models\best.pt",
        key="model-tester-model-path",
        help="Path to a trusted Ultralytics YOLO .pt weights file on this machine.",
    )
    confidence_threshold = st.slider(
        "Confidence threshold for report metrics",
        min_value=0.01,
        max_value=0.99,
        value=0.25,
        step=0.01,
        key="model-tester-confidence",
        help="The suggested threshold is optimized separately for each class.",
    )
    st.caption(
        "Expected dataset layout: images/ and labels/ with matching relative "
        "paths. Each annotation row must be normalized YOLO format: "
        "class_id x_center y_center width height."
    )
    trusted_model = st.checkbox(
        "I trust this model file and its source.",
        key="model-tester-trusted-model",
        help="Only load checkpoints you trust; serialized model files may contain code.",
    )
    if not trusted_model:
        st.warning("Confirm that the model file is trusted before running evaluation.")
        return
    if not dataset_path:
        return

    def evaluate_selected_dataset(progress: ProgressCallback) -> VideoResult:
        return _evaluate_dataset(
            dataset_path,
            model_path,
            confidence_threshold,
            progress,
        )

    _process_button(
        "Evaluate model",
        evaluate_selected_dataset,
    )


def _percentage(value: float | None) -> str:
    return "N/A" if value is None else f"{value * 100:.1f}%"


def _show_evaluation_report(result: VideoResult) -> None:
    try:
        report: dict[str, Any] = json.loads(result.data.decode("utf-8"))
        summary = report["summary"]
        classes = report["classes"]
        confusion_matrix = report["confusion_matrix"]
        false_cases = report["false_cases"]
    except (UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError) as exc:
        st.error(f"Could not display the model report: {exc}")
        _show_result(result)
        return

    st.success(
        f"Evaluation complete: {summary['images']} images, "
        f"{summary['ground_truth_objects']} labeled objects."
    )
    st.caption(
        f"Model: {report['model']} · Confidence threshold: "
        f"{summary['confidence_threshold']:.2f} · IoU threshold: "
        f"{summary['iou_threshold']:.2f}"
    )
    metric_columns = st.columns(4)
    metric_columns[0].metric("Precision", _percentage(summary["precision"]))
    metric_columns[1].metric("Recall / detection rate", _percentage(summary["recall"]))
    metric_columns[2].metric("F1 score", _percentage(summary["f1"]))
    metric_columns[3].metric("mAP@0.50", _percentage(summary["map50"]))
    st.caption(
        f"True positives: {summary['true_positives']} · False detections: "
        f"{summary['false_positives']} · Missed detections: "
        f"{summary['false_negatives']}"
    )

    if classes:
        st.subheader("Per-class performance")
        st.bar_chart(
            classes,
            x="class_name",
            y="detection_percentage",
            x_label="Class",
            y_label="Detection percentage (recall)",
        )
        st.bar_chart(
            classes,
            x="class_name",
            y=["precision", "recall", "f1", "ap50"],
            x_label="Class",
            y_label="Score",
        )
        class_rows = [
            {
                "Class": item["class_name"],
                "Ground truth": item["ground_truth_objects"],
                "Predictions": item["predictions_at_threshold"],
                "True positives": item["true_positives"],
                "False detections": item["false_positives"],
                "Missed detections": item["false_negatives"],
                "Precision": _percentage(item["precision"]),
                "Detection % (recall)": _percentage(item["recall"]),
                "F1": _percentage(item["f1"]),
                "AP@0.50": _percentage(item["ap50"]),
                "Suggested confidence": (
                    f"{item['suggested_confidence']:.2f}"
                    if item["suggested_confidence"] is not None
                    else "N/A (no correct detections)"
                ),
            }
            for item in classes
        ]
        st.dataframe(class_rows, hide_index=True, use_container_width=True)
        st.caption(
            "Suggested confidence maximizes per-class F1 on this same labeled "
            "dataset; validate thresholds on a separate validation set before "
            "using them in production."
        )

    matrix_labels = confusion_matrix["labels"]
    matrix_values = confusion_matrix["values"]
    if matrix_labels and matrix_values:
        st.subheader("Confusion matrix (IoU ≥ 0.50)")
        st.caption("Rows are actual classes; columns are predicted classes.")
        matrix_rows = [
            {
                "Actual / Predicted": label,
                **{
                    column: matrix_values[row_index][column_index]
                    for column_index, column in enumerate(matrix_labels)
                },
            }
            for row_index, label in enumerate(matrix_labels)
        ]
        st.dataframe(matrix_rows, hide_index=True, use_container_width=True)

    st.subheader("False detections and missed detections")
    false_detections = [
        {
            "Image": item["image"],
            "Class": item["class"],
            "Confidence": (
                f"{item['confidence']:.3f}"
                if item["confidence"] is not None
                else ""
            ),
            "Box (left, top, right, bottom)": ", ".join(
                f"{coordinate:.1f}" for coordinate in item["box_xyxy"]
            ),
        }
        for item in false_cases
        if item["type"] == "False detection"
    ]
    missed_detections = [
        {
            "Image": item["image"],
            "Class": item["class"],
            "Box (left, top, right, bottom)": ", ".join(
                f"{coordinate:.1f}" for coordinate in item["box_xyxy"]
            ),
        }
        for item in false_cases
        if item["type"] == "Missed detection"
    ]
    false_tab, missed_tab = st.tabs(
        [
            f"False detections ({len(false_detections)})",
            f"Missed detections ({len(missed_detections)})",
        ]
    )
    with false_tab:
        if false_detections:
            st.dataframe(false_detections, hide_index=True, use_container_width=True)
        else:
            st.info("No false detections at the selected confidence threshold.")
    with missed_tab:
        if missed_detections:
            st.dataframe(missed_detections, hide_index=True, use_container_width=True)
        else:
            st.info("No missed detections at the selected confidence threshold.")
    _show_result(result)


@st.fragment(run_every=0.25)
def render_model_tester_page() -> None:
    _poll_video_job()
    if st.button("← Back to AI", key="model-tester-page-back"):
        _request_back_to_category()
        st.rerun()

    st.title("✨  Model tester")
    st.divider()
    st.caption("Evaluate an Ultralytics YOLO object detector on labeled images.")
    st.warning(
        "Only use trusted .pt model files. Model checkpoints can contain "
        "serialized code and may require substantial CPU, GPU, and memory."
    )
    _render_processing_progress()
    _render_model_tester()

    result = st.session_state.get("video_tool_result")
    if (
        isinstance(result, VideoResult)
        and st.session_state.get("video_processing_tool") == "Model tester"
    ):
        st.divider()
        _show_evaluation_report(result)
