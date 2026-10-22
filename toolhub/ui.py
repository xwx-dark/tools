"""Streamlit presentation layer for the tools hub."""

import streamlit as st

from toolhub.catalog import CATEGORIES, Category
from toolhub.audio_ui import AUDIO_TOOL_NAMES, render_audio_tool_page
from toolhub.annotator_ui import ANNOTATOR_TOOL_NAMES, render_annotator_page
from toolhub.documents_ui import DOCUMENT_TOOL_NAMES, render_document_tool_page
from toolhub.data_ui import DATA_TOOL_NAMES, render_data_tool_page
from toolhub.developer_ui import DEVELOPER_TOOL_NAMES, render_developer_tool_page
from toolhub.files_ui import FILE_TOOL_NAMES, render_file_tool_page
from toolhub.image_ui import IMAGE_TOOL_NAMES, render_image_tool_page
from toolhub.model_tester_ui import MODEL_TESTER_TOOL_NAMES, render_model_tester_page
from toolhub.world_annotator_ui import (
    WORLD_ANNOTATOR_TOOL_NAMES,
    render_world_annotator_page,
)
from toolhub.video_ui import (
    VIDEO_TOOL_NAMES,
    _open_tool,
    handle_category_change,
    render_video_navigation_warning,
    render_video_tool_page,
)


def _render_tool_cards(category: Category, search_query: str) -> None:
    matching_tools = [
        tool
        for tool in category.tools
        if search_query in f"{tool.name} {tool.description}".casefold()
    ]

    if not matching_tools:
        st.info("No tools match your search in this category.")
        return

    columns_per_row = 2
    for start in range(0, len(matching_tools), columns_per_row):
        columns = st.columns(columns_per_row)
        for column, tool in zip(
            columns, matching_tools[start : start + columns_per_row]
        ):
            with column:
                with st.container(border=True):
                    st.subheader(tool.name)
                    st.write(tool.description)
                    available_tools = {
                        "Video": VIDEO_TOOL_NAMES,
                        "Audio": AUDIO_TOOL_NAMES,
                        "Image": IMAGE_TOOL_NAMES,
                        "Documents": DOCUMENT_TOOL_NAMES,
                        "Files": FILE_TOOL_NAMES,
                        "Data": DATA_TOOL_NAMES,
                        "Developer": DEVELOPER_TOOL_NAMES,
                        "AI": (
                            MODEL_TESTER_TOOL_NAMES
                            + ANNOTATOR_TOOL_NAMES
                            + WORLD_ANNOTATOR_TOOL_NAMES
                        ),
                    }
                    if tool.name in available_tools.get(category.name, ()):
                        st.button(
                            "Use tool",
                            key=f"{category.name}-{tool.name}",
                            on_click=_open_tool,
                            args=(tool.name, category.name),
                            type="primary",
                            use_container_width=True,
                        )
                    else:
                        st.button(
                            "Coming soon",
                            key=f"{category.name}-{tool.name}",
                            disabled=True,
                            use_container_width=True,
                        )


def run_app() -> None:
    """Render the tools hub landing page and category navigation."""
    st.set_page_config(
        page_title="Everyday Tools",
        page_icon="🧰",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    with st.sidebar:
        st.title("🧰 Everyday Tools")
        st.caption("A growing collection of useful utilities.")
        st.divider()

        selected_name = st.radio(
            "Categories",
            options=[category.name for category in CATEGORIES],
            key="selected_category",
            on_change=handle_category_change,
            format_func=lambda name: (
                f"{next(category.icon for category in CATEGORIES if category.name == name)}  "
                f"{name}"
            ),
            label_visibility="collapsed",
        )
        search_query = st.text_input(
            "Find a tool",
            placeholder="Search this category...",
        ).strip().casefold()

        st.divider()
        st.caption(
            "Video, audio, image, document, file, data, developer, and AI tools are available. "
            "Other categories are planned."
        )

    selected_category = next(
        category for category in CATEGORIES if category.name == selected_name
    )

    if st.session_state.get("active_tool"):
        category = st.session_state.get("active_tool_category")
        if category == "Audio":
            render_audio_tool_page()
        elif category == "Image":
            render_image_tool_page()
        elif category == "Documents":
            render_document_tool_page()
        elif category == "Files":
            render_file_tool_page()
        elif category == "Data":
            render_data_tool_page()
        elif category == "Developer":
            render_developer_tool_page()
        elif category == "AI":
            active_tool = st.session_state.get("active_tool")
            if active_tool == "Label applier":
                render_annotator_page()
            elif active_tool == "Annotator":
                render_world_annotator_page()
            else:
                render_model_tester_page()
        else:
            render_video_tool_page()
        if st.session_state.get("pending_video_navigation") is not None:
            render_video_navigation_warning()
        return

    st.title(f"{selected_category.icon}  {selected_category.name}")
    st.write(selected_category.description)
    status = (
        "available"
        if selected_category.name
        in {
            "Video",
            "Audio",
            "Image",
            "Documents",
            "Files",
            "Data",
            "Developer",
            "AI",
        }
        else "planned"
    )
    st.caption(f"{len(selected_category.tools)} tools {status}")
    st.divider()
    _render_tool_cards(selected_category, search_query)
