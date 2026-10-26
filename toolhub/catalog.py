"""Data-only catalog of tool categories and future tool placeholders."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Tool:
    name: str
    description: str


@dataclass(frozen=True)
class Category:
    name: str
    icon: str
    description: str
    tools: tuple[Tool, ...]


CATEGORIES = (
    Category(
        name="Video",
        icon="🎬",
        description="Everyday helpers for working with video and clips.",
        tools=(
            Tool("Video compressor", "Reduce video file size."),
            Tool("Video converter", "Convert between common video formats."),
            Tool("Video trimmer", "Trim a clip to the section you need."),
            Tool("GIF maker", "Turn a video clip into an animated GIF."),
            Tool("Thumbnail extractor", "Save a frame as a video thumbnail."),
        ),
    ),
    Category(
        name="Audio",
        icon="🎧",
        description="Simple utilities for audio files and recordings.",
        tools=(
            Tool("Audio converter", "Convert between common audio formats."),
            Tool("Audio trimmer", "Keep only the part of a recording you need."),
            Tool("Volume normalizer", "Bring audio levels closer to a consistent volume."),
            Tool("Audio merger", "Combine multiple audio files into one track."),
        ),
    ),
    Category(
        name="Image",
        icon="🖼️",
        description="Quick tools for common image tasks.",
        tools=(
            Tool("Image resizer", "Resize an image to specific dimensions."),
            Tool("Image converter", "Convert between common image formats."),
            Tool("Image compressor", "Reduce image file size."),
            Tool("Image cropper", "Crop an image to a selected area."),
            Tool("Color picker", "Inspect colors and get their values."),
        ),
    ),
    Category(
        name="Text",
        icon="📝",
        description="Handy helpers for formatting and inspecting text.",
        tools=(
            Tool("Text formatter", "Clean up whitespace and text formatting."),
            Tool("Word and character counter", "Count words, characters, and lines."),
            Tool("Case converter", "Convert text between common letter cases."),
            Tool("Text diff", "Compare two pieces of text."),
        ),
    ),
    Category(
        name="Documents",
        icon="📄",
        description="Practical utilities for PDFs and text documents.",
        tools=(
            Tool("PDF merger", "Combine documents into a single PDF."),
            Tool("PDF splitter", "Separate selected pages into new documents."),
            Tool("PDF to images", "Export document pages as image files."),
            Tool("Document converter", "Convert plain text files to PDF and PDF text to plain text."),
        ),
    ),
    Category(
        name="Files",
        icon="🗂️",
        description="Utilities for organizing and inspecting files.",
        tools=(
            Tool("File renamer", "Batch-rename uploaded files and download them together."),
            Tool("Archive manager", "Create ZIP archives or inspect ZIP and TAR archives."),
            Tool("Duplicate finder", "Identify identical uploaded files by content."),
            Tool("File size analyzer", "Compare uploaded file sizes and export a report."),
        ),
    ),
    Category(
        name="Data",
        icon="📊",
        description="Small helpers for structured data and tables.",
        tools=(
            Tool("CSV viewer", "Preview uploaded CSV data and download a bounded sample."),
            Tool("JSON formatter", "Validate and pretty-print a JSON document."),
            Tool("CSV to JSON", "Convert CSV rows into a JSON array of records."),
            Tool("Data table converter", "Convert tabular data between CSV and TSV."),
        ),
    ),
    Category(
        name="Developer",
        icon="💻",
        description="Convenient utilities for everyday development tasks.",
        tools=(
            Tool("JSON validator", "Check whether JSON is valid."),
            Tool("Base64 encoder", "Encode or decode Base64 text."),
            Tool("URL encoder", "Encode or decode URL components."),
            Tool("Hash generator", "Generate a checksum from input."),
        ),
    ),
    Category(
        name="AI",
        icon="✨",
        description="Tools for exploring and testing AI models.",
        tools=(
            Tool(
                "Model tester",
                "Evaluate an Ultralytics YOLO detector against labeled images.",
            ),
            Tool(
                "Model detector",
                "Run an Ultralytics YOLO model on an image, video, or image folder.",
            ),
            Tool(
                "Label applier",
                "Preview labeled images with bounding boxes and browse the dataset.",
            ),
            Tool(
                "Annotator",
                "Generate YOLO labels from custom classes with YOLO-World or Ollama.",
            ),
        ),
    ),
)
