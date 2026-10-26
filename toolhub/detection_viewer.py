"""Shared image preview and class-count presentation for AI result dialogs."""

import base64
from html import escape
from io import BytesIO
from typing import Sequence

import streamlit as st
from PIL import Image

from toolhub.model_tester import BoxRecord


def render_zoomable_image(
    image: Image.Image,
    alt_text: str,
    height: int = 520,
) -> None:
    """Display an explicitly sized preview inside an independently scrollable area."""
    image_data = BytesIO()
    image.save(image_data, format="PNG")
    encoded_image = base64.b64encode(image_data.getvalue()).decode("ascii")
    escaped_alt_text = escape(alt_text, quote=True)
    st.html(
        f"""
        <div role="region" aria-label="{escaped_alt_text}" tabindex="0"
             style="height:{height}px;overflow:auto;">
          <img src="data:image/png;base64,{encoded_image}" alt="{escaped_alt_text}"
               style="display:block;width:{image.width}px;max-width:none;height:auto;">
        </div>
        """,
        width="stretch",
    )


def render_class_summary(
    annotations: Sequence[BoxRecord],
    class_names: dict[int, str],
    section_title: str = "Classes in this image",
) -> None:
    """Show total and per-class annotation counts below a preview."""
    counts: dict[int, int] = {}
    for annotation in annotations:
        counts[annotation.class_id] = counts.get(annotation.class_id, 0) + 1

    with st.container(border=True):
        st.markdown(f"#### {section_title}")
        if not counts:
            st.caption("No classes or detections in this image.")
            return

        st.metric("Total detections", len(annotations))
        st.dataframe(
            [
                {
                    "Class": (
                        f"{class_names.get(class_id, f'Class {class_id}')} "
                        f"[{class_id}]"
                    ),
                    "Detections": count,
                }
                for class_id, count in sorted(counts.items())
            ],
            hide_index=True,
            width="stretch",
        )
