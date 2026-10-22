"""Pillow-backed image operations used by the Streamlit tool pages."""

from collections.abc import Callable
from io import BytesIO
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from toolhub.video import VideoProcessingError, VideoResult


ProgressCallback = Callable[[float], None]

_FORMAT_DETAILS = {
    "JPEG": ("jpg", "image/jpeg"),
    "PNG": ("png", "image/png"),
    "WEBP": ("webp", "image/webp"),
}


def _open_image(input_data: bytes) -> Image.Image:
    try:
        with Image.open(BytesIO(input_data)) as image:
            image.load()
            return image.copy()
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise VideoProcessingError(
            "Could not open this image. Try a supported, undamaged image file."
        ) from exc


def get_image_dimensions(input_data: bytes) -> tuple[int, int]:
    """Return the validated width and height of an uploaded image."""
    image = _open_image(input_data)
    return image.width, image.height


def _format_details(output_format: str) -> tuple[str, str]:
    details = _FORMAT_DETAILS.get(output_format.upper())
    if details is None:
        raise ValueError("Choose JPEG, PNG, or WebP as the output format.")
    return details


def _save_image(
    image: Image.Image,
    output_format: str,
    quality: int,
    optimize: bool = True,
) -> bytes:
    output_format = output_format.upper()
    _format_details(output_format)
    options: dict[str, object] = {"optimize": optimize}

    if output_format == "JPEG":
        if image.mode in {"RGBA", "LA"} or "transparency" in image.info:
            rgba_image = image.convert("RGBA")
            background = Image.new("RGB", rgba_image.size, "white")
            background.paste(rgba_image, mask=rgba_image.getchannel("A"))
            image = background
        else:
            image = image.convert("RGB")
        options["quality"] = quality
    elif output_format == "WEBP":
        if image.mode not in {"RGB", "RGBA"}:
            image = image.convert("RGBA" if "A" in image.getbands() else "RGB")
        options["quality"] = quality

    buffer = BytesIO()
    try:
        image.save(buffer, format=output_format, **options)
    except (OSError, ValueError) as exc:
        raise VideoProcessingError(
            f"Could not save this image as {output_format}."
        ) from exc
    return buffer.getvalue()


def _result(
    input_name: str,
    suffix: str,
    output_format: str,
    data: bytes,
) -> VideoResult:
    extension, mime_type = _format_details(output_format)
    filename = f"{Path(input_name).stem}_{suffix}.{extension}"
    return VideoResult(data, filename, mime_type)


def resize_image(
    input_data: bytes,
    input_name: str,
    width: int,
    height: int,
    output_format: str,
    quality: int = 90,
    maintain_aspect: bool = True,
    progress: ProgressCallback | None = None,
) -> VideoResult:
    """Resize an image to the chosen dimensions."""
    if width < 1 or height < 1:
        raise ValueError("Image width and height must be greater than zero.")
    if not 1 <= quality <= 100:
        raise ValueError("Choose an image quality between 1 and 100.")
    if progress:
        progress(0.1)
    image = _open_image(input_data)
    if maintain_aspect:
        scale = min(width / image.width, height / image.height)
        output_size = (
            max(1, round(image.width * scale)),
            max(1, round(image.height * scale)),
        )
        image = image.resize(output_size, Image.Resampling.LANCZOS)
    else:
        image = image.resize((width, height), Image.Resampling.LANCZOS)
    if progress:
        progress(0.8)
    data = _save_image(image, output_format, quality)
    if progress:
        progress(1.0)
    return _result(input_name, "resized", output_format, data)


def convert_image(
    input_data: bytes,
    input_name: str,
    output_format: str,
    quality: int = 90,
    progress: ProgressCallback | None = None,
) -> VideoResult:
    """Convert an image to JPEG, PNG, or WebP."""
    if not 1 <= quality <= 100:
        raise ValueError("Choose an image quality between 1 and 100.")
    if progress:
        progress(0.1)
    image = _open_image(input_data)
    if progress:
        progress(0.6)
    data = _save_image(image, output_format, quality)
    if progress:
        progress(1.0)
    return _result(input_name, "converted", output_format, data)


def compress_image(
    input_data: bytes,
    input_name: str,
    output_format: str,
    quality: int,
    progress: ProgressCallback | None = None,
) -> VideoResult:
    """Compress an image using format-specific encoding options."""
    if not 1 <= quality <= 100:
        raise ValueError("Choose an image quality between 1 and 100.")
    if progress:
        progress(0.1)
    image = _open_image(input_data)
    if progress:
        progress(0.6)
    data = _save_image(image, output_format, quality, optimize=True)
    if progress:
        progress(1.0)
    return _result(input_name, "compressed", output_format, data)


def crop_image(
    input_data: bytes,
    input_name: str,
    left: int,
    top: int,
    width: int,
    height: int,
    output_format: str,
    quality: int = 90,
    progress: ProgressCallback | None = None,
) -> VideoResult:
    """Crop an image to the selected rectangle."""
    if left < 0 or top < 0 or width < 1 or height < 1:
        raise ValueError("Crop position must be non-negative and size must be positive.")
    if not 1 <= quality <= 100:
        raise ValueError("Choose an image quality between 1 and 100.")
    if progress:
        progress(0.1)
    image = _open_image(input_data)
    if left + width > image.width or top + height > image.height:
        raise ValueError("The crop area must fit within the image dimensions.")
    cropped = image.crop((left, top, left + width, top + height))
    if progress:
        progress(0.8)
    data = _save_image(cropped, output_format, quality)
    if progress:
        progress(1.0)
    return _result(input_name, "cropped", output_format, data)


def pick_color(
    input_data: bytes,
    input_name: str,
    x: int,
    y: int,
    progress: ProgressCallback | None = None,
) -> VideoResult:
    """Read the pixel color at the given image coordinates."""
    if x < 0 or y < 0:
        raise ValueError("Pixel coordinates must be non-negative.")
    if progress:
        progress(0.1)
    image = _open_image(input_data)
    if x >= image.width or y >= image.height:
        raise ValueError("The selected pixel must be inside the image.")
    red, green, blue = image.convert("RGB").getpixel((x, y))
    hex_color = f"#{red:02X}{green:02X}{blue:02X}"
    if progress:
        progress(1.0)
    return VideoResult(
        f"{hex_color}\nRGB({red}, {green}, {blue})\nX: {x}, Y: {y}\n".encode(),
        f"{Path(input_name).stem}_color.txt",
        "text/plain",
    )
