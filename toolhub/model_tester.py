"""Evaluation helpers for Ultralytics YOLO object-detection models."""

from collections import defaultdict
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any

from PIL import Image

from toolhub.video import ProgressCallback, VideoProcessingError, VideoResult


class ModelEvaluationError(VideoProcessingError):
    """An actionable failure while evaluating an object-detection model."""


_IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
_BOX_LINE_DESCRIPTION = "class_id x_center y_center width height"
_MIN_INFERENCE_CONFIDENCE = 0.001


@dataclass(frozen=True)
class BoxRecord:
    image: str
    class_id: int
    box: tuple[float, float, float, float]


@dataclass(frozen=True)
class DetectionRecord(BoxRecord):
    confidence: float


def _report_progress(progress: ProgressCallback | None, value: float) -> None:
    if progress:
        progress(min(1.0, max(0.0, value)))


def load_yolo_dataset(
    dataset_path: str | Path,
) -> tuple[list[tuple[Path, str]], dict[str, list[BoxRecord]], int]:
    root = Path(dataset_path).expanduser()
    if not root.exists() or not root.is_dir():
        raise ModelEvaluationError("The truth-base path must be an accessible folder.")
    images_dir = root / "images"
    labels_dir = root / "labels"
    if not images_dir.is_dir() or not labels_dir.is_dir():
        raise ModelEvaluationError(
            "The truth-base folder must contain 'images' and 'labels' folders."
        )

    image_paths = sorted(
        (
            path
            for path in images_dir.rglob("*")
            if path.is_file() and path.suffix.casefold() in _IMAGE_SUFFIXES
        ),
        key=lambda path: path.relative_to(images_dir).as_posix().casefold(),
    )
    if not image_paths:
        raise ModelEvaluationError("No supported images were found in the images folder.")

    samples: list[tuple[Path, str]] = []
    ground_truth: dict[str, list[BoxRecord]] = {}
    object_count = 0
    for image_path in image_paths:
        relative = image_path.relative_to(images_dir)
        label_path = (labels_dir / relative).with_suffix(".txt")
        if not label_path.is_file():
            raise ModelEvaluationError(
                f"Missing label file for '{relative.as_posix()}'."
            )
        try:
            with Image.open(image_path) as image:
                image_width, image_height = image.size
                image.verify()
        except (OSError, ValueError) as exc:
            raise ModelEvaluationError(
                f"Could not read image '{relative.as_posix()}': {exc}"
            ) from exc
        if image_width < 1 or image_height < 1:
            raise ModelEvaluationError(
                f"Image '{relative.as_posix()}' has invalid dimensions."
            )

        image_key = relative.as_posix()
        records = _parse_annotation_file(
            label_path, image_key, image_width, image_height
        )
        samples.append((image_path, image_key))
        ground_truth[image_key] = records
        object_count += len(records)
    return samples, ground_truth, object_count


def _parse_annotation_file(
    annotation_path: Path,
    image_key: str,
    image_width: int,
    image_height: int,
) -> list[BoxRecord]:
    records = []
    try:
        lines = annotation_path.read_text(encoding="utf-8-sig").splitlines()
    except (OSError, UnicodeError) as exc:
        raise ModelEvaluationError(
            f"Could not read annotation for '{image_key}': {exc}"
        ) from exc
    for line_number, line in enumerate(lines, start=1):
        parts = line.split()
        if not parts:
            continue
        if len(parts) != 5:
            raise ModelEvaluationError(
                f"Annotation '{annotation_path.name}', line {line_number}, must contain "
                f"{_BOX_LINE_DESCRIPTION}."
            )
        try:
            class_value = float(parts[0])
            class_id = int(class_value)
            x_center, y_center, width, height = map(float, parts[1:])
        except ValueError as exc:
            raise ModelEvaluationError(
                f"Annotation '{annotation_path.name}', line {line_number}, "
                "contains a non-numeric value."
            ) from exc
        values = (class_value, x_center, y_center, width, height)
        if (
            not all(math.isfinite(value) for value in values)
            or class_value != class_id
            or class_id < 0
            or not 0 <= x_center <= 1
            or not 0 <= y_center <= 1
            or not 0 < width <= 1
            or not 0 < height <= 1
            or x_center - width / 2 < -1e-6
            or x_center + width / 2 > 1 + 1e-6
            or y_center - height / 2 < -1e-6
            or y_center + height / 2 > 1 + 1e-6
        ):
            raise ModelEvaluationError(
                f"Annotation '{annotation_path.name}', line {line_number}, "
                "has an invalid class id or normalized bounding box."
            )
        left = max(0.0, (x_center - width / 2) * image_width)
        top = max(0.0, (y_center - height / 2) * image_height)
        right = min(float(image_width), (x_center + width / 2) * image_width)
        bottom = min(float(image_height), (y_center + height / 2) * image_height)
        records.append(BoxRecord(image_key, class_id, (left, top, right, bottom)))
    return records


def _class_names(model: Any) -> dict[int, str]:
    raw_names = model.names
    if isinstance(raw_names, dict):
        names = {int(class_id): str(name) for class_id, name in raw_names.items()}
    elif isinstance(raw_names, (list, tuple)):
        names = {index: str(name) for index, name in enumerate(raw_names)}
    else:
        raise ModelEvaluationError("The selected model does not expose class names.")
    if not names:
        raise ModelEvaluationError("The selected model does not define any classes.")
    return names


def _intersection_over_union(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> float:
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union > 0 else 0.0


def _class_prediction_outcomes(
    predictions: list[DetectionRecord],
    ground_truth: dict[str, list[BoxRecord]],
    class_id: int,
    iou_threshold: float,
) -> tuple[list[tuple[float, bool]], int]:
    matched: dict[str, set[int]] = defaultdict(set)
    class_predictions = sorted(
        (prediction for prediction in predictions if prediction.class_id == class_id),
        key=lambda prediction: prediction.confidence,
        reverse=True,
    )
    outcomes = []
    for prediction in class_predictions:
        candidates = ground_truth.get(prediction.image, [])
        best_index = -1
        best_iou = iou_threshold
        for index, target in enumerate(candidates):
            if target.class_id != class_id or index in matched[prediction.image]:
                continue
            overlap = _intersection_over_union(prediction.box, target.box)
            if overlap >= best_iou:
                best_iou = overlap
                best_index = index
        if best_index >= 0:
            matched[prediction.image].add(best_index)
        outcomes.append((prediction.confidence, best_index >= 0))
    total_targets = sum(
        1
        for targets in ground_truth.values()
        for target in targets
        if target.class_id == class_id
    )
    return outcomes, total_targets


def _average_precision(
    outcomes: list[tuple[float, bool]],
    total_targets: int,
) -> float:
    if total_targets == 0:
        return 0.0
    cumulative_tp = 0
    cumulative_fp = 0
    recall_values = []
    precision_values = []
    for _, is_true_positive in outcomes:
        cumulative_tp += int(is_true_positive)
        cumulative_fp += int(not is_true_positive)
        recall_values.append(cumulative_tp / total_targets)
        precision_values.append(cumulative_tp / (cumulative_tp + cumulative_fp))
    return sum(
        max(
            (
                precision
                for recall, precision in zip(recall_values, precision_values)
                if recall >= recall_point
            ),
            default=0.0,
        )
        for recall_point in (index / 100 for index in range(101))
    ) / 101


def _confusion_matrix(
    predictions: list[DetectionRecord],
    ground_truth: dict[str, list[BoxRecord]],
    class_ids: list[int],
    iou_threshold: float,
    confidence_threshold: float,
) -> list[list[int]]:
    matrix_index = {class_id: index for index, class_id in enumerate(class_ids)}
    background_index = len(class_ids)
    matrix = [
        [0 for _ in range(len(class_ids) + 1)]
        for _ in range(len(class_ids) + 1)
    ]
    predictions_by_image: dict[str, list[DetectionRecord]] = defaultdict(list)
    for prediction in predictions:
        if prediction.confidence >= confidence_threshold:
            predictions_by_image[prediction.image].append(prediction)

    for image_key, targets in ground_truth.items():
        image_predictions = sorted(
            predictions_by_image.get(image_key, []),
            key=lambda prediction: prediction.confidence,
            reverse=True,
        )
        candidate_pairs = sorted(
            (
                (
                    _intersection_over_union(target.box, prediction.box),
                    target_index,
                    prediction_index,
                )
                for target_index, target in enumerate(targets)
                for prediction_index, prediction in enumerate(image_predictions)
                if _intersection_over_union(target.box, prediction.box) >= iou_threshold
            ),
            reverse=True,
        )
        matched_targets: set[int] = set()
        matched_predictions: set[int] = set()
        for _, target_index, prediction_index in candidate_pairs:
            if (
                target_index in matched_targets
                or prediction_index in matched_predictions
            ):
                continue
            target = targets[target_index]
            prediction = image_predictions[prediction_index]
            matrix[matrix_index[target.class_id]][
                matrix_index[prediction.class_id]
            ] += 1
            matched_targets.add(target_index)
            matched_predictions.add(prediction_index)
        for target_index, target in enumerate(targets):
            if target_index not in matched_targets:
                matrix[matrix_index[target.class_id]][background_index] += 1
        for prediction_index, prediction in enumerate(image_predictions):
            if prediction_index not in matched_predictions:
                matrix[background_index][matrix_index[prediction.class_id]] += 1
    return matrix


def calculate_detection_metrics(
    class_names: dict[int, str],
    ground_truth: dict[str, list[BoxRecord]],
    predictions: list[DetectionRecord],
    confidence_threshold: float = 0.25,
    iou_threshold: float = 0.5,
) -> dict[str, Any]:
    """Calculate per-class precision/recall/F1, AP50, and best-F1 thresholds."""
    if not 0 <= confidence_threshold <= 1 or not 0 < iou_threshold <= 1:
        raise ModelEvaluationError("Confidence and IoU thresholds are out of range.")
    actual_class_ids = {
        target.class_id
        for targets in ground_truth.values()
        for target in targets
    }
    predicted_class_ids = {prediction.class_id for prediction in predictions}
    unknown_ids = (actual_class_ids | predicted_class_ids) - set(class_names)
    if unknown_ids:
        raise ModelEvaluationError(
            "Dataset class IDs do not match the model class names: "
            + ", ".join(str(class_id) for class_id in sorted(unknown_ids))
        )
    class_ids = sorted(actual_class_ids | predicted_class_ids)
    classes = []
    total_tp = total_fp = total_fn = 0
    for class_id in class_ids:
        outcomes, target_count = _class_prediction_outcomes(
            predictions, ground_truth, class_id, iou_threshold
        )
        selected_outcomes = [
            is_true_positive
            for confidence, is_true_positive in outcomes
            if confidence >= confidence_threshold
        ]
        tp = sum(selected_outcomes)
        fp = len(selected_outcomes) - tp
        fn = target_count - tp
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        ap50 = _average_precision(outcomes, target_count)
        suggested_confidence = None
        best_threshold_f1 = 0.0
        best_threshold_precision = 0.0
        if target_count and outcomes:
            cumulative_tp = cumulative_fp = 0
            outcome_index = 0
            while outcome_index < len(outcomes):
                threshold = outcomes[outcome_index][0]
                while (
                    outcome_index < len(outcomes)
                    and outcomes[outcome_index][0] == threshold
                ):
                    if outcomes[outcome_index][1]:
                        cumulative_tp += 1
                    else:
                        cumulative_fp += 1
                    outcome_index += 1
                threshold_fn = target_count - cumulative_tp
                threshold_precision = (
                    cumulative_tp / (cumulative_tp + cumulative_fp)
                    if cumulative_tp + cumulative_fp
                    else 0.0
                )
                threshold_recall = (
                    cumulative_tp / (cumulative_tp + threshold_fn)
                    if cumulative_tp + threshold_fn
                    else 0.0
                )
                threshold_f1 = (
                    2 * threshold_precision * threshold_recall
                    / (threshold_precision + threshold_recall)
                    if threshold_precision + threshold_recall
                    else 0.0
                )
                if (
                    suggested_confidence is None
                    or threshold_f1 > best_threshold_f1
                    or (
                        math.isclose(threshold_f1, best_threshold_f1)
                        and threshold_precision > best_threshold_precision
                    )
                ):
                    suggested_confidence = threshold
                    best_threshold_f1 = threshold_f1
                    best_threshold_precision = threshold_precision
            if best_threshold_f1 == 0:
                suggested_confidence = None
        classes.append(
            {
                "class_id": class_id,
                "class_name": class_names[class_id],
                "ground_truth_objects": target_count,
                "predictions_at_threshold": tp + fp,
                "true_positives": tp,
                "false_positives": fp,
                "false_negatives": fn,
                "precision": precision,
                "recall": recall,
                "detection_percentage": recall * 100,
                "f1": f1,
                "ap50": ap50,
                "suggested_confidence": suggested_confidence,
                "f1_at_suggested_confidence": (
                    best_threshold_f1 if suggested_confidence is not None else None
                ),
            }
        )
        total_tp += tp
        total_fp += fp
        total_fn += fn

    total_precision = total_tp / (total_tp + total_fp) if total_tp + total_fp else 0.0
    total_recall = total_tp / (total_tp + total_fn) if total_tp + total_fn else 0.0
    total_f1 = (
        2 * total_precision * total_recall / (total_precision + total_recall)
        if total_precision + total_recall
        else 0.0
    )
    evaluated_classes = [item for item in classes if item["ground_truth_objects"] > 0]
    map50 = (
        sum(item["ap50"] for item in evaluated_classes) / len(evaluated_classes)
        if evaluated_classes
        else 0.0
    )
    return {
        "summary": {
            "images": len(ground_truth),
            "classes": len(classes),
            "ground_truth_objects": sum(
                len(targets) for targets in ground_truth.values()
            ),
            "predictions_at_threshold": total_tp + total_fp,
            "true_positives": total_tp,
            "false_positives": total_fp,
            "false_negatives": total_fn,
            "confidence_threshold": confidence_threshold,
            "iou_threshold": iou_threshold,
            "precision": total_precision,
            "recall": total_recall,
            "f1": total_f1,
            "map50": map50,
        },
        "classes": classes,
        "confusion_matrix": {
            "labels": [
                f"{class_names[class_id]} [{class_id}]" for class_id in class_ids
            ]
            + ["Background"],
            "values": _confusion_matrix(
                predictions,
                ground_truth,
                class_ids,
                iou_threshold,
                confidence_threshold,
            ),
        },
    }


def evaluate_yolo_model(
    dataset_path: str | Path,
    model_path: str | Path,
    confidence_threshold: float = 0.25,
    progress: ProgressCallback | None = None,
) -> VideoResult:
    """Run local Ultralytics YOLO inference and return a JSON evaluation report."""
    confidence_threshold = float(confidence_threshold)
    if not 0 <= confidence_threshold <= 1:
        raise ModelEvaluationError("Confidence threshold must be between 0 and 1.")
    weights = Path(model_path).expanduser()
    if not weights.is_file() or weights.suffix.casefold() != ".pt":
        raise ModelEvaluationError(
            "Choose an accessible Ultralytics YOLO .pt weights file."
        )
    samples, ground_truth, ground_truth_count = load_yolo_dataset(dataset_path)
    if ground_truth_count == 0:
        raise ModelEvaluationError(
            "The dataset contains no annotated objects, so detection metrics cannot be calculated."
        )
    _report_progress(progress, 0.02)
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ModelEvaluationError(
            "Ultralytics is unavailable. Install project dependencies with "
            "'python -m pip install -r requirements.txt'."
        ) from exc
    try:
        model = YOLO(str(weights))
    except Exception as exc:
        raise ModelEvaluationError(
            f"Could not load the YOLO model: {type(exc).__name__}: {exc}"
        ) from exc
    class_names = _class_names(model)
    for targets in ground_truth.values():
        for target in targets:
            if target.class_id not in class_names:
                raise ModelEvaluationError(
                    f"Annotation class ID {target.class_id} is not defined by the model."
                )

    predictions: list[DetectionRecord] = []
    ignored_invalid_predictions = 0
    invalid_prediction_images: set[str] = set()
    for index, (image_path, image_key) in enumerate(samples):
        try:
            results = model.predict(
                source=str(image_path),
                conf=_MIN_INFERENCE_CONFIDENCE,
                verbose=False,
                save=False,
            )
            result = results[0]
            boxes = result.boxes
            if boxes is not None and len(boxes):
                coordinates = boxes.xyxy.cpu().tolist()
                class_ids = boxes.cls.cpu().tolist()
                confidences = boxes.conf.cpu().tolist()
                if not (
                    len(coordinates) == len(class_ids) == len(confidences)
                ):
                    raise ModelEvaluationError(
                        f"Inference returned inconsistent detections for '{image_key}'."
                    )
                for box, class_id, confidence in zip(
                    coordinates, class_ids, confidences
                ):
                    try:
                        class_value = float(class_id)
                        box_values = tuple(float(value) for value in box)
                        confidence_value = float(confidence)
                    except (TypeError, ValueError, OverflowError):
                        ignored_invalid_predictions += 1
                        invalid_prediction_images.add(image_key)
                        continue
                    if (
                        not math.isfinite(class_value)
                        or not class_value.is_integer()
                        or not math.isfinite(confidence_value)
                        or not 0 <= confidence_value <= 1
                        or len(box_values) != 4
                        or not all(math.isfinite(value) for value in box_values)
                        or box_values[2] <= box_values[0]
                        or box_values[3] <= box_values[1]
                    ):
                        ignored_invalid_predictions += 1
                        invalid_prediction_images.add(image_key)
                        continue
                    class_id = int(class_value)
                    if class_id not in class_names:
                        raise ModelEvaluationError(
                            f"Model prediction returned unknown class ID {class_id}."
                        )
                    predictions.append(
                        DetectionRecord(
                            image_key,
                            class_id,
                            box_values,
                            confidence_value,
                        )
                    )
        except ModelEvaluationError:
            raise
        except Exception as exc:
            raise ModelEvaluationError(
                f"Inference failed for '{image_key}': {type(exc).__name__}: {exc}"
            ) from exc
        _report_progress(progress, 0.05 + 0.75 * (index + 1) / len(samples))

    metrics = calculate_detection_metrics(
        class_names,
        ground_truth,
        predictions,
        confidence_threshold=confidence_threshold,
        iou_threshold=0.5,
    )
    metrics["model"] = weights.name
    metrics["dataset"] = str(Path(dataset_path).expanduser().resolve())
    metrics["summary"]["ignored_invalid_predictions"] = (
        ignored_invalid_predictions
    )
    metrics["summary"]["invalid_prediction_images"] = sorted(
        invalid_prediction_images,
        key=str.casefold,
    )
    metrics["classes"] = [
        {
            **item,
            "suggested_confidence": (
                round(item["suggested_confidence"], 4)
                if item["suggested_confidence"] is not None
                else None
            ),
        }
        for item in metrics["classes"]
    ]

    false_cases = []
    matched_targets_by_image: dict[str, set[int]] = defaultdict(set)
    for class_id in sorted(class_names):
        matched: dict[str, set[int]] = defaultdict(set)
        for prediction in sorted(
            (
                prediction
                for prediction in predictions
                if prediction.class_id == class_id
                and prediction.confidence >= confidence_threshold
            ),
            key=lambda item: item.confidence,
            reverse=True,
        ):
            targets = ground_truth.get(prediction.image, [])
            best_index = -1
            best_iou = 0.5
            for target_index, target in enumerate(targets):
                if (
                    target.class_id != class_id
                    or target_index in matched[prediction.image]
                ):
                    continue
                overlap = _intersection_over_union(prediction.box, target.box)
                if overlap >= best_iou:
                    best_iou = overlap
                    best_index = target_index
            if best_index >= 0:
                matched[prediction.image].add(best_index)
                matched_targets_by_image[prediction.image].add(best_index)
            else:
                false_cases.append(
                    {
                        "image": prediction.image,
                        "type": "False detection",
                        "class": class_names[class_id],
                        "confidence": prediction.confidence,
                        "box_xyxy": list(prediction.box),
                    }
                )
    for image_key, targets in ground_truth.items():
        for target_index, target in enumerate(targets):
            if target_index not in matched_targets_by_image[image_key]:
                false_cases.append(
                    {
                        "image": image_key,
                        "type": "Missed detection",
                        "class": class_names[target.class_id],
                        "confidence": None,
                        "box_xyxy": list(target.box),
                    }
                )
    metrics["false_cases"] = false_cases
    _report_progress(progress, 0.98)
    report = (json.dumps(metrics, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    _report_progress(progress, 1.0)
    return VideoResult(
        report,
        "model-test-report.json",
        "application/json",
    )
