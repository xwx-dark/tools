"""Page-based user interfaces for the image tools."""

import streamlit as st

from toolhub.image import (
    compress_image,
    convert_image,
    crop_image,
    get_image_dimensions,
    pick_color,
    resize_image,
)
from toolhub.video import VideoProcessingError, VideoResult
from toolhub.video_ui import (
    _poll_video_job,
    _process_button,
    _render_processing_progress,
    _request_back_to_category,
    _show_result,
)


IMAGE_TOOL_NAMES = (
    "Image resizer",
    "Image converter",
    "Image compressor",
    "Image cropper",
    "Color picker",
)

_IMAGE_FILE_TYPES = [
    "png",
    "jpg",
    "jpeg",
    "webp",
    "gif",
    "bmp",
    "tif",
    "tiff",
]
_OUTPUT_FORMATS = ["PNG", "JPEG", "WebP"]


def _upload_image(tool_key: str):
    return st.file_uploader(
        "Choose an image",
        type=_IMAGE_FILE_TYPES,
        key=f"image-{tool_key}-upload",
        help="Image processing runs on the machine running this app.",
    )


def _validated_image(uploaded_file) -> tuple[bytes, str, int, int] | None:
    if uploaded_file is None:
        return None
    image_data = uploaded_file.getvalue()
    try:
        width, height = get_image_dimensions(image_data)
    except VideoProcessingError as exc:
        st.error(str(exc))
        return None
    return image_data, uploaded_file.name, width, height


def _quality_control(key: str, *, disabled: bool = False) -> int:
    return st.slider(
        "Quality",
        min_value=1,
        max_value=100,
        value=90,
        help="Higher values retain more detail and usually create larger files.",
        key=key,
        disabled=disabled,
    )


def _render_resizer() -> None:
    uploaded_file = _upload_image("resizer")
    image = _validated_image(uploaded_file)
    if image is None:
        return
    image_data, input_name, original_width, original_height = image
    st.caption(f"Original dimensions: {original_width} × {original_height} px")
    maintain_aspect = st.checkbox(
        "Maintain aspect ratio", value=True, key="image-resizer-aspect"
    )
    width = st.number_input(
        "Output width (pixels)",
        min_value=1,
        max_value=20000,
        value=original_width,
        step=1,
        key="image-resizer-width",
    )
    height = st.number_input(
        "Output height (pixels)",
        min_value=1,
        max_value=20000,
        value=original_height,
        step=1,
        key="image-resizer-height",
        disabled=maintain_aspect,
    )
    output_format = st.selectbox(
        "Output format", _OUTPUT_FORMATS, key="image-resizer-format"
    )
    quality = _quality_control(
        "image-resizer-quality", disabled=output_format == "PNG"
    )
    _process_button(
        "Resize image",
        lambda progress: resize_image(
            image_data,
            input_name,
            width,
            height,
            output_format,
            quality,
            maintain_aspect,
            progress,
        ),
    )


def _render_converter() -> None:
    uploaded_file = _upload_image("converter")
    image = _validated_image(uploaded_file)
    if image is None:
        return
    image_data, input_name, _, _ = image
    output_format = st.selectbox(
        "Convert to", _OUTPUT_FORMATS, key="image-converter-format"
    )
    quality = _quality_control(
        "image-converter-quality", disabled=output_format == "PNG"
    )
    _process_button(
        "Convert image",
        lambda progress: convert_image(
            image_data, input_name, output_format, quality, progress
        ),
    )


def _render_compressor() -> None:
    uploaded_file = _upload_image("compressor")
    image = _validated_image(uploaded_file)
    if image is None:
        return
    image_data, input_name, width, height = image
    st.caption(f"Image dimensions: {width} × {height} px")
    output_format = st.selectbox(
        "Compressed format", _OUTPUT_FORMATS, key="image-compressor-format"
    )
    quality = _quality_control(
        "image-compressor-quality", disabled=output_format == "PNG"
    )
    st.caption(
        "PNG uses lossless optimization; JPEG and WebP use the selected quality."
    )
    _process_button(
        "Compress image",
        lambda progress: compress_image(
            image_data, input_name, output_format, quality, progress
        ),
    )


def _render_cropper() -> None:
    uploaded_file = _upload_image("cropper")
    image = _validated_image(uploaded_file)
    if image is None:
        return
    image_data, input_name, image_width, image_height = image
    st.caption(f"Image dimensions: {image_width} × {image_height} px")
    left = st.number_input(
        "Left position (pixels)",
        min_value=0,
        max_value=image_width - 1,
        value=0,
        key="image-crop-left",
    )
    top = st.number_input(
        "Top position (pixels)",
        min_value=0,
        max_value=image_height - 1,
        value=0,
        key="image-crop-top",
    )
    width = st.number_input(
        "Crop width (pixels)",
        min_value=1,
        max_value=image_width,
        value=image_width,
        key="image-crop-width",
    )
    height = st.number_input(
        "Crop height (pixels)",
        min_value=1,
        max_value=image_height,
        value=image_height,
        key="image-crop-height",
    )
    output_format = st.selectbox(
        "Output format", _OUTPUT_FORMATS, key="image-crop-format"
    )
    quality = _quality_control(
        "image-crop-quality", disabled=output_format == "PNG"
    )
    _process_button(
        "Crop image",
        lambda progress: crop_image(
            image_data,
            input_name,
            left,
            top,
            width,
            height,
            output_format,
            quality,
            progress,
        ),
    )


def _render_color_picker() -> None:
    uploaded_file = _upload_image("color-picker")
    image = _validated_image(uploaded_file)
    if image is None:
        return
    image_data, input_name, width, height = image
    st.caption(
        f"Image dimensions: {width} × {height} px. Coordinates use the original image."
    )
    st.image(image_data, caption=input_name, use_container_width=True)

    x = st.number_input(
        "X coordinate (pixels)", min_value=0, max_value=width - 1, value=0,
        key="image-color-x",
    )
    y = st.number_input(
        "Y coordinate (pixels)", min_value=0, max_value=height - 1, value=0,
        key="image-color-y",
    )
    _process_button(
        "Pick color",
        lambda progress: pick_color(image_data, input_name, x, y, progress),
    )

    result = st.session_state.get("video_tool_result")
    if (
        isinstance(result, VideoResult)
        and st.session_state.get("video_processing_tool") == "Color picker"
    ):
        try:
            hex_color = result.data.decode("ascii").splitlines()[0]
            st.color_picker("Picked color", value=hex_color, disabled=True)
            st.code(result.data.decode("ascii"))
            _show_result(result)
        except (UnicodeDecodeError, IndexError, ValueError) as exc:
            st.error(f"Could not display the picked color: {exc}")


@st.fragment(run_every=0.25)
def render_image_tool_page() -> None:
    _poll_video_job()
    if st.button("← Back to Image", key="image-page-back"):
        _request_back_to_category()
        st.rerun()

    tool_name = st.session_state.get("active_tool")
    st.title(f"🖼️  {tool_name or 'Image tool'}")
    st.divider()
    st.caption("Image processing runs on the machine running this app.")
    _render_processing_progress()

    tools = {
        "Image resizer": _render_resizer,
        "Image converter": _render_converter,
        "Image compressor": _render_compressor,
        "Image cropper": _render_cropper,
        "Color picker": _render_color_picker,
    }
    render_tool = tools.get(tool_name)
    if render_tool is None:
        st.error("This image tool is not available.")
        return
    render_tool()

    result = st.session_state.get("video_tool_result")
    if (
        isinstance(result, VideoResult)
        and st.session_state.get("video_processing_tool") == tool_name
        and tool_name != "Color picker"
    ):
        st.divider()
        _show_result(result)
