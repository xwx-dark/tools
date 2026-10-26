# Everyday Tools

A Streamlit application for a collection of general-purpose utilities. Video,
audio, image, document, file, data, developer, and AI tools are implemented;
remaining categories are browsable placeholders.

## Video tools

Choose **Video** in the sidebar, then open a tool to compress, convert, trim,
create a GIF, or extract a thumbnail. Each tool opens as a page with a Back
button and a live progress bar while processing; the process button stays
disabled until the job finishes or reports an error. Results are offered as
downloads. FFmpeg and ffprobe must be installed and available on PATH on the
machine running the app. Video operations are handled by
`toolhub/video.py`; page controls live in `toolhub/video_ui.py`.

## Audio tools

Choose **Audio** in the sidebar to convert audio (MP3, WAV, AAC, FLAC, OGG, or
M4A), trim a clip, normalize perceived loudness, or merge multiple recordings.
Each tool opens as a page with a Back button, live FFmpeg progress, and a
download when processing completes.

## Image tools

Choose **Image** in the sidebar to resize, convert, or compress an image, crop
an image by pixel coordinates, or pick a pixel color and inspect its hex/RGB
values. Image outputs support PNG, JPEG, and WebP.

## Document tools

Choose **Documents** to merge PDF files, split selected pages into separate
PDFs, export PDF pages as PNG or JPEG images, or convert between PDF and plain
text. The converter extracts text from PDFs and creates simple PDFs from text;
it does not preserve the layout or formatting of rich-text documents.

## File tools

Choose **Files** to batch-rename uploaded files, create ZIP archives, inspect
ZIP/TAR/TAR.GZ archive contents without extracting them, find identical files
by SHA-256, or review file sizes and download a CSV report. File tools operate
on files uploaded to the current page; they do not read arbitrary folders on
the machine running the app.

## Data tools

Choose **Data** to preview CSV data (up to a selected sample size), validate
and pretty-print JSON, convert CSV rows to a JSON array of records, or convert
between CSV and TSV. CSV and TSV inputs are treated as UTF-8 text with a
header row.

## Developer tools

Choose **Developer** to validate and format JSON, encode or decode Base64,
percent-encode or decode URL components, or generate MD5, SHA-1, SHA-256, and
SHA-512 checksums. Hash outputs are checksums and should not be used for
password storage or security-sensitive signing.

## AI tools

Choose **AI → Model tester** to evaluate an Ultralytics YOLO `.pt` object
detector against a labeled image folder selected with the desktop folder picker.
The folder must contain `images/` and `labels/` directories with matching
relative paths. Each label is a YOLO text file containing normalized
`class_id x_center y_center width height` rows, with class IDs matching the
model. Evaluation reports precision, recall, F1, mAP@0.50, per-class detection
percentages, false detections, missed detections, a confusion matrix, and a
per-class confidence suggestion based on best F1 on this dataset. The
suggested thresholds are tuning guidance, not independent validation. Invalid
detections returned by the model are skipped and listed in the evaluation
warning rather than stopping the full dataset evaluation.

Choose **AI → Label applier** and use the desktop folder picker to select the
dataset. After loading the folder, choose **Show results** to open the image
viewer in a modal. The viewer uses a 70/30 image-and-controls layout; hover over
the icon buttons for descriptions, use the left/right arrow keys to navigate,
or use the image-picker button and confirm to jump to a frame. The zoom buttons
change the preview scale. Class names are loaded from a dataset-root
`data.yaml`/`data.yml` (`names`), `classes.txt`, or the Annotator-generated
`annotator_classes.txt`; without a mapping, boxes use their numeric class IDs.
This tool only previews annotations and does not modify the source images or
labels.

Choose **AI → Model detector** to select a trusted Ultralytics YOLO `.pt` model
and detect objects in one image, a video, or a folder of images. The folder
input searches nested folders for supported image files. Choose a confidence
threshold, run detection, and use **Show results** to browse frames or images
with the model's class labels, confidence values, and bounding boxes. A JSON
summary can be downloaded; input files are not modified.

Choose **AI → Annotator** to generate labels with either Ultralytics YOLO-World
or a vision-capable model installed in your local Ollama instance. Select a
dataset folder containing `images/`, enter one custom class per line, and
choose a confidence threshold. For Ollama, start the local Ollama service and
refresh the model list; only models reporting vision capability are offered.
The generated normalized YOLO text files are written under `labels/` with
matching relative paths; a `annotator_classes.txt` file records the class-ID
order. Existing matching labels or class-name metadata require explicit
confirmation before they can be replaced. Ollama's boxes and confidence values
are model estimates; boxes extending beyond image edges are clipped, while
malformed or unusable detections are counted in the report and shown as a
warning. Review those labels before using them for training.
Vision-language models are not dedicated object detectors and may estimate
inaccurate or repeated boxes across different images. Ollama errors mentioning
an unknown `mllama` architecture mean the installed Ollama runtime needs to be
updated; for Llama 3.2 Vision, use an installed `llama3.2-vision` model rather
than the text-only `llama3.2` model.

YOLO-World dependencies can be installed with
`python -m pip install -r requirements.txt` in the environment running
Streamlit. This includes Ultralytics' CLIP package. The first run may download
YOLO-World detector weights and CLIP's ViT-B/32 text weights. You can choose a
local YOLO-World `.pt` file to avoid the detector download; fully offline runs
also need the CLIP package installed and `ViT-B-32.pt` cached under
Ultralytics' configured `weights_dir/clip` folder. The UI explains these
requirements when network access is restricted.

AI tools that read local files use native desktop file/folder pickers and access
the selected files directly without uploading or copying the dataset. Streamlit
must run on the same desktop where the files are accessible; these pickers are
not available in a remote browser-hosted deployment. Model files must also be
accessible to that machine. Only use trusted model files; checkpoints may
contain serialized code. Install the model runtime with
`python -m pip install -r requirements.txt`. Inference uses the runtime's
default device selection and can be slow or memory intensive on CPU or large
datasets.

## Run locally

```powershell
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

The project config allows individual uploads up to 10 GiB. Large uploads need
sufficient memory and temporary storage on the machine running the app.

## Project structure

```text
app.py                 Streamlit entry point
toolhub/
  catalog.py           Category and tool placeholder data
  ui.py                Sidebar navigation and page rendering
  audio.py             FFmpeg-backed audio processing operations
  audio_ui.py          Audio tool pages and controls
  image.py             Pillow-backed image processing operations
  image_ui.py          Image tool pages and controls
  documents.py         PyMuPDF-backed PDF and text processing operations
  documents_ui.py      Document tool pages and controls
  files.py             File renaming, archive, duplicate, and size operations
  files_ui.py          File tool pages and controls
  data.py              CSV and JSON processing operations
  data_ui.py           Data tool pages and controls
  developer.py         JSON, Base64, URL, and hash operations
  developer_ui.py      Developer tool pages and controls
  annotator.py          YOLO annotation loading and image rendering
  annotator_ui.py       AI annotation browsing page
  local_folder_picker.py Native desktop file and folder pickers
  model_detector.py     Local YOLO inference for image, video, and folder inputs
  model_detector_ui.py  AI detection page and results viewer
  model_tester.py       YOLO inference and detection metrics
  model_tester_ui.py    AI model test page and report charts
  video.py             FFmpeg-backed video processing operations
  video_ui.py          Video tool pages, progress, and downloads
video_compressor.py    Existing standalone video-compression script
```

Add or update categories and tools in `toolhub/catalog.py`. Keep Streamlit
navigation in `toolhub/ui.py`; add future processing code in a focused service
module and its controls in the matching UI module.
