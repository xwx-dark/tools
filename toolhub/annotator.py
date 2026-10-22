"""Load and render image annotations in YOLO text format."""

import colorsys
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont
import yaml

from toolhub.model_tester import BoxRecord, ModelEvaluationError, load_yolo_dataset


class AnnotatorError(ModelEvaluationError):
    """An actionable failure while loading an annotated image."""


def _read_class_names(dataset_path: str | Path) -> dict[int, str]:
    root = Path(dataset_path).expanduser()
    generated_names = _read_names_file(root / "annotator_classes.txt")
    if generated_names:
        return generated_names
    for filename in ("data.yaml", "data.yml"):
        names_path = root / filename
        if not names_path.is_file():
            continue
        try:
            contents: Any = yaml.safe_load(
                names_path.read_text(encoding="utf-8-sig")
            )
        except (OSError, UnicodeError, yaml.YAMLError) as exc:
            raise AnnotatorError(
                f"Could not read class names from '{filename}': {exc}"
            ) from exc
        if not isinstance(contents, dict):
            raise AnnotatorError(f"'{filename}' must contain a YAML mapping.")
        names = contents.get("names")
        if isinstance(names, list):
            entries = enumerate(names)
        elif isinstance(names, dict):
            entries = names.items()
        elif names is None:
            return _read_names_file(root / "classes.txt")
        else:
            raise AnnotatorError(
                f"The 'names' value in '{filename}' must be a list or mapping."
            )
        class_names = {}
        for class_id, name in entries:
            try:
                numeric_id = int(class_id)
            except (TypeError, ValueError) as exc:
                raise AnnotatorError(
                    f"The 'names' mapping in '{filename}' contains a "
                    "non-numeric class ID."
                ) from exc
            if numeric_id < 0 or not isinstance(name, str) or not name.strip():
                raise AnnotatorError(
                    f"The 'names' value in '{filename}' must map non-negative class IDs "
                    "to non-empty names."
                )
            class_names[numeric_id] = name.strip()
        return class_names
    return _read_names_file(root / "classes.txt")


def _read_names_file(classes_path: Path) -> dict[int, str]:
    if not classes_path.is_file():
        return {}
    try:
        names = classes_path.read_text(encoding="utf-8-sig").splitlines()
    except (OSError, UnicodeError) as exc:
        raise AnnotatorError(
            f"Could not read class names from 'classes.txt': {exc}"
        ) from exc
    return {
        class_id: name.strip()
        for class_id, name in enumerate(names)
        if name.strip()
    }


def load_annotator_dataset(
    dataset_path: str | Path,
) -> tuple[list[tuple[Path, str]], dict[str, list[BoxRecord]], dict[int, str]]:
    samples, annotations, _ = load_yolo_dataset(dataset_path)
    return samples, annotations, _read_class_names(dataset_path)


def render_annotated_image(
    image_path: str | Path,
    annotations: list[BoxRecord],
    zoom: float = 1.0,
    class_names: dict[int, str] | None = None,
) -> Image.Image:
    """Return an annotated image scaled relative to a 1000-pixel preview width."""
    if not 0.5 <= zoom <= 2.0:
        raise AnnotatorError("Zoom must be between 50% and 200%.")
    try:
        with Image.open(image_path) as source:
            image = source.convert("RGB")
    except (OSError, ValueError) as exc:
        raise AnnotatorError(f"Could not open image '{image_path}': {exc}") from exc

    draw = ImageDraw.Draw(image)
    line_width = max(2, round(min(image.size) * 0.003))
    font = ImageFont.load_default()
    for annotation in annotations:
        hue = (annotation.class_id * 0.61803398875) % 1.0
        red, green, blue = colorsys.hsv_to_rgb(hue, 0.85, 1.0)
        color = (round(red * 255), round(green * 255), round(blue * 255))
        draw.rectangle(annotation.box, outline=color, width=line_width)
        class_name = (class_names or {}).get(
            annotation.class_id, f"Class {annotation.class_id}"
        )
        label = f"{class_name} [{annotation.class_id}]"
        _, top, _, bottom = draw.textbbox((0, 0), label, font=font)
        label_top = max(0, round(annotation.box[1]) - (bottom - top) - 6)
        draw.rectangle(
            (
                round(annotation.box[0]),
                label_top,
                round(annotation.box[0]) + (draw.textlength(label, font=font)) + 6,
                label_top + (bottom - top) + 4,
            ),
            fill=color,
        )
        draw.text(
            (round(annotation.box[0]) + 3, label_top + 2 - top),
            label,
            fill="black",
            font=font,
        )

    target_width = max(1, round(min(image.width, 1000) * zoom))
    if target_width != image.width:
        target_height = max(1, round(image.height * target_width / image.width))
        image = image.resize((target_width, target_height), Image.Resampling.LANCZOS)
    return image
