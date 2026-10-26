"""Run local Ultralytics object detection on images, videos, or image folders."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any

from PIL import Image

from toolhub.model_tester import (
    DetectionRecord,
    ModelEvaluationError,
    _class_names,
)
from toolhub.video import ProgressCallback, VideoResult


_IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
_VIDEO_SUFFIXES = {".avi", ".m4v", ".mkv", ".mov", ".mp4", ".webm"}


@dataclass(frozen=True)
class DetectionFrame:
    image_path: str
    image_key: str
    frame_number: int | None
    detections: list[DetectionRecord]


@dataclass(frozen=True)
class ModelDetectorResult(VideoResult):
    model_name: str
    input_type: str
    input_path: str
    class_names: dict[int, str]
    frames: list[DetectionFrame]


def _report_progress(progress: ProgressCallback | None, value: float) -> None:
    if progress:
        progress(min(1.0, max(0.0, value)))


def _image_paths(folder: Path) -> list[Path]:
    if not folder.is_dir():
        raise ModelEvaluationError("Choose an accessible folder containing images.")
    paths = sorted(
        (
            path
            for path in folder.rglob("*")
            if path.is_file() and path.suffix.casefold() in _IMAGE_SUFFIXES
        ),
        key=lambda path: path.relative_to(folder).as_posix().casefold(),
    )
    if not paths:
        raise ModelEvaluationError("No supported images were found in the selected folder.")
    return paths


def _load_model(model_path: Path) -> Any:
    if not model_path.is_file() or model_path.suffix.casefold() != ".pt":
        raise ModelEvaluationError("Choose an accessible Ultralytics YOLO .pt model file.")
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ModelEvaluationError(
            "Ultralytics is unavailable. Install project dependencies with "
            "'python -m pip install -r requirements.txt'."
        ) from exc
    try:
        model = YOLO(str(model_path))
    except Exception as exc:
        raise ModelEvaluationError(
            f"Could not load the YOLO model: {type(exc).__name__}: {exc}"
        ) from exc
    return model


def _detections_from_result(
    result: Any,
    image_key: str,
    class_names: dict[int, str],
) -> list[DetectionRecord]:
    boxes = result.boxes
    if boxes is None or not len(boxes):
        return []
    coordinates = boxes.xyxy.cpu().tolist()
    class_ids = boxes.cls.cpu().tolist()
    confidences = boxes.conf.cpu().tolist()
    if not (len(coordinates) == len(class_ids) == len(confidences)):
        raise ModelEvaluationError(
            f"Inference returned inconsistent detections for '{image_key}'."
        )

    detections: list[DetectionRecord] = []
    for box, class_value, confidence_value in zip(
        coordinates, class_ids, confidences
    ):
        numeric_class_id = float(class_value)
        confidence = float(confidence_value)
        box_values = tuple(float(value) for value in box)
        if (
            not math.isfinite(numeric_class_id)
            or not numeric_class_id.is_integer()
            or not math.isfinite(confidence)
            or not 0 <= confidence <= 1
            or len(box_values) != 4
            or not all(math.isfinite(value) for value in box_values)
            or box_values[2] <= box_values[0]
            or box_values[3] <= box_values[1]
        ):
            raise ModelEvaluationError(
                f"Inference returned an invalid detection for '{image_key}'."
            )
        class_id = int(numeric_class_id)
        if class_id not in class_names:
            raise ModelEvaluationError(
                f"Model prediction returned unknown class ID {class_id}."
            )
        detections.append(
            DetectionRecord(image_key, class_id, box_values, confidence)
        )
    return detections


def _predict_image(
    model: Any,
    image_path: Path,
    image_key: str,
    confidence_threshold: float,
    class_names: dict[int, str],
) -> DetectionFrame:
    try:
        results = model.predict(
            source=str(image_path),
            conf=confidence_threshold,
            verbose=False,
            save=False,
        )
        detections = _detections_from_result(results[0], image_key, class_names)
    except ModelEvaluationError:
        raise
    except Exception as exc:
        raise ModelEvaluationError(
            f"Inference failed for '{image_key}': {type(exc).__name__}: {exc}"
        ) from exc
    return DetectionFrame(str(image_path), image_key, None, detections)


def detect_with_yolo_model(
    input_type: str,
    input_path: str | Path,
    model_path: str | Path,
    confidence_threshold: float = 0.25,
    progress: ProgressCallback | None = None,
) -> ModelDetectorResult:
    """Detect objects in one image, a video, or all images under a folder."""
    if input_type not in {"Image", "Video", "Path"}:
        raise ModelEvaluationError("Choose Image, Video, or Path as the input type.")
    confidence_threshold = float(confidence_threshold)
    if not math.isfinite(confidence_threshold) or not 0 <= confidence_threshold <= 1:
        raise ModelEvaluationError("Confidence threshold must be between 0 and 1.")

    source = Path(input_path).expanduser()
    weights = Path(model_path).expanduser()
    if not source.exists():
        raise ModelEvaluationError("Choose an accessible input file or folder.")
    if input_type == "Image":
        if not source.is_file() or source.suffix.casefold() not in _IMAGE_SUFFIXES:
            raise ModelEvaluationError("Choose a supported image file.")
        image_paths = [source]
    elif input_type == "Video":
        if not source.is_file() or source.suffix.casefold() not in _VIDEO_SUFFIXES:
            raise ModelEvaluationError("Choose a supported video file.")
        image_paths = []
    else:
        image_paths = _image_paths(source)

    model = _load_model(weights)
    class_names = _class_names(model)
    frames: list[DetectionFrame] = []

    if input_type == "Video":
        try:
            import cv2
        except ImportError as exc:
            raise ModelEvaluationError(
                "OpenCV is unavailable. Install project dependencies with "
                "'python -m pip install -r requirements.txt'."
            ) from exc
        capture = cv2.VideoCapture(str(source))
        if not capture.isOpened():
            capture.release()
            raise ModelEvaluationError("Could not open the selected video.")
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        capture.release()
        try:
            results = model.predict(
                source=str(source),
                conf=confidence_threshold,
                verbose=False,
                save=False,
                stream=True,
            )
            for frame_number, result in enumerate(results):
                image_key = f"{source.name} · frame {frame_number + 1}"
                detections = _detections_from_result(
                    result, image_key, class_names
                )
                frames.append(
                    DetectionFrame(
                        str(source),
                        image_key,
                        frame_number,
                        detections,
                    )
                )
                progress_value = (
                    (frame_number + 1) / total_frames
                    if total_frames > 0
                    else 0.95 * (frame_number + 1) / (frame_number + 2)
                )
                _report_progress(progress, progress_value)
        except ModelEvaluationError:
            raise
        except Exception as exc:
            raise ModelEvaluationError(
                f"Video inference failed: {type(exc).__name__}: {exc}"
            ) from exc
        if not frames:
            raise ModelEvaluationError("The selected video contains no readable frames.")
    else:
        for index, image_path in enumerate(image_paths):
            image_key = (
                image_path.name
                if input_type == "Image"
                else image_path.relative_to(source).as_posix()
            )
            frames.append(
                _predict_image(
                    model,
                    image_path,
                    image_key,
                    confidence_threshold,
                    class_names,
                )
            )
            _report_progress(progress, (index + 1) / len(image_paths))

    detection_count = sum(len(frame.detections) for frame in frames)
    summary = {
        "model": weights.name,
        "input_type": input_type,
        "input": str(source.resolve()),
        "frames": len(frames),
        "detections": detection_count,
    }
    return ModelDetectorResult(
        data=json.dumps(summary, indent=2).encode("utf-8"),
        filename="model-detection-summary.json",
        mime_type="application/json",
        model_name=weights.name,
        input_type=input_type,
        input_path=str(source.resolve()),
        class_names=class_names,
        frames=frames,
    )


def read_video_frame(video_path: str | Path, frame_number: int) -> Image.Image:
    """Read a single zero-based video frame as a PIL image for the results viewer."""
    try:
        import cv2
    except ImportError as exc:
        raise ModelEvaluationError(
            "OpenCV is unavailable. Install project dependencies with "
            "'python -m pip install -r requirements.txt'."
        ) from exc
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        capture.release()
        raise ModelEvaluationError(f"Could not open video '{video_path}'.")
    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        success, frame = capture.read()
    finally:
        capture.release()
    if not success or frame is None:
        raise ModelEvaluationError(
            f"Could not read video frame {frame_number + 1} from '{video_path}'."
        )
    return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
