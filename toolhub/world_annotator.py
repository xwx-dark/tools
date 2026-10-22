"""Generate YOLO-format annotations with vision models."""

from __future__ import annotations

import base64
import json
import importlib
import math
import os
from pathlib import Path
import tempfile
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from PIL import Image
from toolhub.model_tester import ModelEvaluationError
from toolhub.video import ProgressCallback, VideoResult


_IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
_CLASS_NAMES_FILENAME = "annotator_classes.txt"
_OLLAMA_BASE_URL = "http://localhost:11434"


def find_dataset_images(dataset_path: str | Path) -> list[Path]:
    root = Path(dataset_path).expanduser()
    images_dir = root / "images"
    if not root.is_dir():
        raise ModelEvaluationError("Choose an accessible dataset folder.")
    if not images_dir.is_dir():
        raise ModelEvaluationError(
            "The selected dataset folder must contain an 'images' folder."
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
        raise ModelEvaluationError(
            "No supported images were found in the images folder."
        )
    return image_paths


def existing_output_files(dataset_path: str | Path) -> list[Path]:
    root = Path(dataset_path).expanduser()
    images = find_dataset_images(root)
    labels_dir = root / "labels"
    outputs = [
        (labels_dir / image.relative_to(root / "images")).with_suffix(".txt")
        for image in images
    ]
    if len(set(outputs)) != len(outputs):
        raise ModelEvaluationError(
            "Multiple images map to the same YOLO label filename. "
            "Rename images with duplicate names but different extensions."
        )
    existing = [path for path in outputs if path.exists()]
    class_names_path = root / _CLASS_NAMES_FILENAME
    if class_names_path.exists():
        existing.append(class_names_path)
    return existing


def _atomic_write(path: Path, contents: str) -> None:
    temporary_path: Path | None = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_file.write(contents)
            temporary_path = Path(temporary_file.name)
        os.replace(temporary_path, path)
    except OSError as exc:
        if temporary_path is not None:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError as cleanup_error:
                exc.add_note(
                    f"Could not remove temporary file '{temporary_path}': "
                    f"{cleanup_error}"
                )
        raise ModelEvaluationError(
            f"Could not write annotation output '{path}': {exc}"
        ) from exc


def _ollama_request(
    endpoint: str, payload: dict[str, Any] | None = None, timeout: float = 10.0
) -> dict[str, Any]:
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    request = Request(
        f"{_OLLAMA_BASE_URL}{endpoint}",
        data=body,
        headers={"Content-Type": "application/json"} if body is not None else {},
        method="GET" if body is None else "POST",
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            result: Any = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace").strip()
        if (
            endpoint == "/api/chat"
            and "unknown model architecture" in detail.casefold()
            and "mllama" in detail.casefold()
        ):
            raise ModelEvaluationError(
                "The Ollama runtime cannot load this model's mllama architecture. "
                "Update Ollama to the latest version, restart the Ollama service, "
                "and pull the selected vision model again. If you intended to use "
                "Llama 3.2 Vision, select an installed llama3.2-vision model; the "
                "plain llama3.2 text model is not a vision model."
            ) from exc
        raise ModelEvaluationError(
            f"Ollama API request to {endpoint} failed with HTTP {exc.code}: "
            f"{detail or exc.reason}"
        ) from exc
    except URLError as exc:
        raise ModelEvaluationError(
            "Could not connect to Ollama at http://localhost:11434. Start the "
            "Ollama application/service and make sure its local API is available."
        ) from exc
    except OSError as exc:
        raise ModelEvaluationError(
            f"Ollama local API request to {endpoint} failed: {exc}"
        ) from exc
    except (TimeoutError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ModelEvaluationError(
            f"Ollama returned an invalid or timed-out response from {endpoint}: {exc}"
        ) from exc
    if not isinstance(result, dict):
        raise ModelEvaluationError(f"Ollama returned an invalid response from {endpoint}.")
    return result


def list_ollama_vision_models() -> list[str]:
    """Return locally installed Ollama models whose API advertises vision support."""
    tags_response = _ollama_request("/api/tags")
    models = tags_response.get("models")
    if not isinstance(models, list):
        raise ModelEvaluationError("Ollama's model list response has no models array.")

    vision_models: list[str] = []
    for entry in models:
        if not isinstance(entry, dict):
            raise ModelEvaluationError("Ollama returned an invalid model entry.")
        name = entry.get("name") or entry.get("model")
        if not isinstance(name, str) or not name:
            raise ModelEvaluationError("Ollama returned a model without a name.")
        details = _ollama_request(
            "/api/show", {"model": name, "verbose": False}
        )
        capabilities = details.get("capabilities", [])
        if not isinstance(capabilities, list):
            raise ModelEvaluationError(
                f"Ollama returned invalid capability data for model '{name}'."
            )
        if "vision" in capabilities:
            vision_models.append(name)
    return vision_models


def _ollama_detections(
    model_name: str,
    image_path: Path,
    class_names: list[str],
    image_width: int,
    image_height: int,
) -> list[dict[str, Any]]:
    try:
        image_data = image_path.read_bytes()
    except OSError as exc:
        raise ModelEvaluationError(
            f"Could not read image '{image_path.name}' for Ollama: {exc}"
        ) from exc

    class_map = {class_id: name for class_id, name in enumerate(class_names)}
    schema = {
        "type": "object",
        "properties": {
            "detections": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "class_id": {"type": "integer"},
                        "x_center": {"type": "number"},
                        "y_center": {"type": "number"},
                        "width": {"type": "number"},
                        "height": {"type": "number"},
                        "confidence": {"type": "number"},
                    },
                    "required": [
                        "class_id",
                        "x_center",
                        "y_center",
                        "width",
                        "height",
                        "confidence",
                    ],
                },
            }
        },
        "required": ["detections"],
    }
    prompt = (
        f"Analyze only the attached image, which is {image_width} pixels wide "
        f"and {image_height} pixels tall. Detect every visible instance "
        "belonging to the provided classes. Estimate each box from the visible "
        "object boundaries in this image, not from a previous image or a typical "
        "object position. "
        "Return image-relative normalized bounding boxes: x_center, y_center, "
        "width, and height must each be between 0 and 1. Keep boxes within the "
        "image boundaries. Use the closest listed "
        "class only; do not invent classes. Return an empty detections list if "
        "no listed objects are visible. Class IDs are: "
        f"{json.dumps(class_map, ensure_ascii=False)}. "
        "Confidence is your estimated probability from 0 to 1."
    )
    image_message = {
        "role": "user",
        "content": prompt,
        "images": [base64.b64encode(image_data).decode("ascii")],
    }
    messages = [image_message]
    parse_error: json.JSONDecodeError | None = None
    parsed: Any = None
    for attempt in range(2):
        response = _ollama_request(
            "/api/chat",
            {
                "model": model_name,
                "messages": messages,
                "stream": False,
                "format": schema,
                "options": {"temperature": 0, "num_predict": 8192},
            },
            timeout=600.0,
        )
        message = response.get("message")
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, str):
            raise ModelEvaluationError(
                f"Ollama model '{model_name}' returned no annotation for "
                f"'{image_path.name}'."
            )
        try:
            parsed = json.loads(content)
            parse_error = None
            break
        except json.JSONDecodeError as exc:
            parse_error = exc
            if attempt == 0:
                messages = [
                    image_message,
                    {"role": "assistant", "content": content},
                    {
                        "role": "user",
                        "content": (
                            "The previous response was not valid JSON. Regenerate "
                            "the detections from the image and return only one "
                            "complete JSON object matching the required schema. "
                            "Do not include commentary or markdown."
                        ),
                    },
                ]
    if parse_error is not None:
        raise ModelEvaluationError(
            f"Ollama model '{model_name}' returned invalid detection JSON for "
            f"'{image_path.name}' after one retry: {parse_error}. Try a different "
            "vision model or reduce the number of detections requested by using "
            "more specific classes."
        ) from parse_error
    detections = parsed.get("detections") if isinstance(parsed, dict) else None
    if not isinstance(detections, list):
        raise ModelEvaluationError(
            f"Ollama model '{model_name}' returned no detections array for "
            f"'{image_path.name}'."
        )
    return detections


def annotate_dataset_with_yolo_world(
    dataset_path: str | Path,
    class_names: list[str],
    confidence_threshold: float = 0.25,
    overwrite_existing: bool = False,
    progress: ProgressCallback | None = None,
    model_path: str | Path | None = None,
    backend: str = "YOLO-World",
    ollama_model: str | None = None,
) -> VideoResult:
    """Annotate each image and save normalized YOLO rows under ``labels/``."""
    root = Path(dataset_path).expanduser().resolve()
    image_paths = find_dataset_images(root)
    if not class_names or any(not name.strip() for name in class_names):
        raise ModelEvaluationError("Enter at least one non-empty class name.")
    if any("\n" in name or "\r" in name for name in class_names):
        raise ModelEvaluationError("Each class name must fit on one line.")
    if len({name.strip().casefold() for name in class_names}) != len(class_names):
        raise ModelEvaluationError("Class names must be unique, ignoring letter case.")
    class_names = [name.strip() for name in class_names]
    if (
        not math.isfinite(confidence_threshold)
        or not 0.0 < confidence_threshold <= 1.0
    ):
        raise ModelEvaluationError("Confidence must be greater than 0 and at most 1.")

    previous_outputs = existing_output_files(root)
    if previous_outputs and not overwrite_existing:
        raise ModelEvaluationError(
            f"{len(previous_outputs)} output file(s) already exist. "
            "Confirm replacing existing annotations before processing."
        )
    if backend not in {"YOLO-World", "Ollama"}:
        raise ModelEvaluationError(f"Unsupported annotation backend '{backend}'.")
    if backend == "Ollama" and not ollama_model:
        raise ModelEvaluationError("Choose an Ollama vision model before annotating.")
    if backend == "YOLO-World" and model_path is not None:
        weights_path = Path(model_path).expanduser()
        if not weights_path.is_file() or weights_path.suffix.casefold() != ".pt":
            raise ModelEvaluationError(
                "Choose an accessible YOLO-World .pt weights file."
            )
    else:
        weights_path = None

    model = None
    if backend == "YOLO-World":
        try:
            try:
                importlib.import_module("clip")
            except ModuleNotFoundError as exc:
                if exc.name == "clip":
                    raise ModelEvaluationError(
                        "YOLO-World requires Ultralytics' CLIP package. Install project "
                        "dependencies with 'python -m pip install -r requirements.txt' "
                        "in the same Python environment that runs Streamlit, then restart "
                        "the app. If this machine has no network, install the CLIP package "
                        "from a connected machine or administrator-provided package source."
                    ) from exc
                raise

            from ultralytics import YOLOWorld

            model = YOLOWorld(str(weights_path)) if weights_path else YOLOWorld()
            model.set_classes(class_names)
        except ModelEvaluationError:
            raise
        except URLError as exc:
            raise ModelEvaluationError(
                "YOLO-World could not download required detector or CLIP weights because "
                f"the host could not be reached ({exc}). Check DNS/network access and "
                "retry. A local YOLO-World .pt file avoids downloading detector weights, "
                "but CLIP's ViT-B/32 weights must also be cached in Ultralytics' "
                "weights_dir/clip folder for fully offline use."
            ) from exc
        except Exception as exc:
            raise ModelEvaluationError(
                f"Could not initialize YOLO-World: {type(exc).__name__}: {exc}. "
                "Check that project dependencies are installed and that required "
                "weights are available locally or downloadable."
            ) from exc

    annotations: list[tuple[Path, str]] = []
    total_detections = 0
    ignored_detections = 0
    for index, image_path in enumerate(image_paths):
        try:
            if backend == "Ollama":
                with Image.open(image_path) as image:
                    width, height = image.size
                detections = _ollama_detections(
                    ollama_model or "", image_path, class_names, width, height
                )
            else:
                if model is None:
                    raise ModelEvaluationError("YOLO-World model was not initialized.")
                prediction_results = model.predict(
                    source=str(image_path),
                    conf=confidence_threshold,
                    verbose=False,
                )
                if not prediction_results:
                    raise ModelEvaluationError(
                        f"YOLO-World returned no result for '{image_path.name}'."
                    )
                result = prediction_results[0]
                width, height = result.orig_shape[1], result.orig_shape[0]
                detections = []
                boxes = result.boxes
                if boxes is not None:
                    xyxy_values = boxes.xyxy.cpu().tolist()
                    class_ids = boxes.cls.cpu().tolist()
                    confidences = boxes.conf.cpu().tolist()
                    if not (
                        len(xyxy_values) == len(class_ids) == len(confidences)
                    ):
                        raise ModelEvaluationError(
                            f"YOLO-World returned inconsistent detections for "
                            f"'{image_path.name}'."
                        )
                    for box, class_id, confidence in zip(
                        xyxy_values, class_ids, confidences
                    ):
                        left = min(float(width), max(0.0, float(box[0])))
                        top = min(float(height), max(0.0, float(box[1])))
                        right = min(float(width), max(0.0, float(box[2])))
                        bottom = min(float(height), max(0.0, float(box[3])))
                        if right <= left or bottom <= top:
                            continue
                        detections.append(
                            {
                                "class_id": int(class_id),
                                "x_center": ((left + right) / 2) / width,
                                "y_center": ((top + bottom) / 2) / height,
                                "width": (right - left) / width,
                                "height": (bottom - top) / height,
                                "confidence": float(confidence),
                            }
                        )
            if width < 1 or height < 1:
                raise ModelEvaluationError(
                    f"{backend} returned invalid image dimensions for "
                    f"'{image_path.name}'."
                )
            rows: list[str] = []
            for detection in detections:
                if not isinstance(detection, dict):
                    ignored_detections += 1
                    continue
                try:
                    class_value = detection["class_id"]
                    class_id = int(class_value)
                    x_center = float(detection["x_center"])
                    y_center = float(detection["y_center"])
                    box_width = float(detection["width"])
                    box_height = float(detection["height"])
                    confidence = float(detection["confidence"])
                except (KeyError, OverflowError, TypeError, ValueError) as exc:
                    ignored_detections += 1
                    continue
                values = (x_center, y_center, box_width, box_height, confidence)
                if (
                    isinstance(class_value, bool)
                    or float(class_value) != class_id
                    or not 0 <= class_id < len(class_names)
                    or not all(math.isfinite(value) for value in values)
                    or not 0 < box_width <= 1
                    or not 0 < box_height <= 1
                    or not 0 <= confidence <= 1
                ):
                    ignored_detections += 1
                    continue
                left = max(0.0, x_center - box_width / 2)
                top = max(0.0, y_center - box_height / 2)
                right = min(1.0, x_center + box_width / 2)
                bottom = min(1.0, y_center + box_height / 2)
                if right <= left or bottom <= top:
                    ignored_detections += 1
                    continue
                if confidence < confidence_threshold:
                    continue
                clipped_x_center = (left + right) / 2
                clipped_y_center = (top + bottom) / 2
                clipped_width = right - left
                clipped_height = bottom - top
                rows.append(
                    f"{class_id} {clipped_x_center:.6f} {clipped_y_center:.6f} "
                    f"{clipped_width:.6f} {clipped_height:.6f}"
                )
            output_path = (
                root / "labels" / image_path.relative_to(root / "images")
            ).with_suffix(".txt")
            annotations.append((output_path, "\n".join(rows) + ("\n" if rows else "")))
            total_detections += len(rows)
        except ModelEvaluationError:
            raise
        except Exception as exc:
            raise ModelEvaluationError(
                f"{backend} inference failed for '{image_path.name}': "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        if progress:
            progress(0.05 + 0.9 * (index + 1) / len(image_paths))

    for output_path, contents in annotations:
        _atomic_write(output_path, contents)
    _atomic_write(root / _CLASS_NAMES_FILENAME, "\n".join(class_names) + "\n")

    summary: dict[str, Any] = {
        "dataset": str(root),
        "images": len(image_paths),
        "detections": total_detections,
        "ignored_detections": ignored_detections,
        "classes": class_names,
        "backend": backend,
        "model": ollama_model if backend == "Ollama" else (
            str(weights_path) if weights_path else "yolov8s-world.pt"
        ),
        "labels_directory": str(root / "labels"),
        "class_names_file": str(root / _CLASS_NAMES_FILENAME),
        "overwritten_files": len(previous_outputs),
    }
    if progress:
        progress(1.0)
    return VideoResult(
        data=json.dumps(summary, indent=2).encode("utf-8"),
        filename="annotation-report.json",
        mime_type="application/json",
    )
