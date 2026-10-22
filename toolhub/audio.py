"""FFmpeg-backed audio operations used by the Streamlit tool pages."""

import json
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Callable

from toolhub.video import VideoProcessingError, VideoResult, _run_ffmpeg


ProgressCallback = Callable[[float], None]

_FORMATS = {
    "mp3": ("libmp3lame", "audio/mpeg"),
    "wav": ("pcm_s16le", "audio/wav"),
    "aac": ("aac", "audio/aac"),
    "flac": ("flac", "audio/flac"),
    "ogg": ("libvorbis", "audio/ogg"),
    "m4a": ("aac", "audio/mp4"),
}


def _probe_audio(input_path: Path) -> float:
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration:stream=codec_type",
                "-of",
                "json",
                str(input_path),
            ],
            capture_output=True,
            check=True,
            text=True,
            timeout=60,
        )
    except FileNotFoundError as exc:
        raise VideoProcessingError(
            "ffprobe was not found. Install FFmpeg and add its bin folder to PATH."
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise VideoProcessingError("Audio inspection timed out.") from exc
    except subprocess.CalledProcessError as exc:
        detail = exc.stderr.strip() or "Unsupported or invalid audio file."
        raise VideoProcessingError(f"Could not read audio metadata: {detail}") from exc

    try:
        metadata = json.loads(result.stdout)
        duration = float(metadata["format"]["duration"])
        has_audio = any(
            stream.get("codec_type") == "audio"
            for stream in metadata.get("streams", [])
        )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise VideoProcessingError(
            "Could not determine the audio duration. Try another audio file."
        ) from exc
    if duration <= 0 or not has_audio:
        raise VideoProcessingError("The uploaded file does not contain usable audio.")
    return duration


def _process_audio(
    inputs: list[tuple[bytes, str]],
    output_name: str,
    mime_type: str,
    operation: Callable[[list[Path], Path, list[float], ProgressCallback | None], None],
    progress: ProgressCallback | None,
) -> VideoResult:
    with TemporaryDirectory(prefix="everyday-tools-audio-") as directory:
        work_dir = Path(directory)
        input_paths = []
        durations = []
        for index, (data, name) in enumerate(inputs):
            suffix = Path(name).suffix.lower() or ".mp3"
            input_path = work_dir / f"input-{index}{suffix}"
            input_path.write_bytes(data)
            input_paths.append(input_path)
            durations.append(_probe_audio(input_path))

        output_path = work_dir / output_name
        operation(input_paths, output_path, durations, progress)
        if not output_path.is_file() or output_path.stat().st_size == 0:
            raise VideoProcessingError(
                "FFmpeg finished without creating an output file."
            )
        output_data = output_path.read_bytes()
    return VideoResult(output_data, output_name, mime_type)


def convert_audio(
    input_data: bytes,
    input_name: str,
    output_format: str,
    progress: ProgressCallback | None = None,
) -> VideoResult:
    """Convert an audio file to a supported format."""
    output_format = output_format.lower().lstrip(".")
    if output_format not in _FORMATS:
        raise ValueError("Choose MP3, WAV, AAC, FLAC, OGG, or M4A.")
    codec, mime_type = _FORMATS[output_format]
    output_name = f"{Path(input_name).stem}_converted.{output_format}"

    def operation(
        paths: list[Path],
        output: Path,
        durations: list[float],
        callback: ProgressCallback | None,
    ) -> None:
        _run_ffmpeg(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(paths[0]),
                "-vn",
                "-map",
                "0:a:0",
                "-c:a",
                codec,
                str(output),
            ],
            durations[0],
            callback,
        )

    return _process_audio(
        [(input_data, input_name)], output_name, mime_type, operation, progress
    )


def trim_audio(
    input_data: bytes,
    input_name: str,
    start_seconds: float,
    end_seconds: float,
    progress: ProgressCallback | None = None,
) -> VideoResult:
    """Trim an audio file between start and end times."""
    if start_seconds < 0 or end_seconds <= start_seconds:
        raise ValueError("End time must be greater than a non-negative start time.")
    output_name = f"{Path(input_name).stem}_trimmed.mp3"

    def operation(
        paths: list[Path],
        output: Path,
        durations: list[float],
        callback: ProgressCallback | None,
    ) -> None:
        if end_seconds > durations[0]:
            raise ValueError("End time cannot be after the end of the audio.")
        _run_ffmpeg(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-ss",
                str(start_seconds),
                "-i",
                str(paths[0]),
                "-t",
                str(end_seconds - start_seconds),
                "-map",
                "0:a:0",
                "-vn",
                "-c:a",
                "libmp3lame",
                "-q:a",
                "2",
                str(output),
            ],
            end_seconds - start_seconds,
            callback,
        )

    return _process_audio(
        [(input_data, input_name)], output_name, "audio/mpeg", operation, progress
    )


def normalize_audio(
    input_data: bytes,
    input_name: str,
    target_lufs: int,
    progress: ProgressCallback | None = None,
) -> VideoResult:
    """Normalize perceived loudness to the selected integrated LUFS target."""
    if not -24 <= target_lufs <= -14:
        raise ValueError("Choose a loudness target between -24 and -14 LUFS.")
    output_name = f"{Path(input_name).stem}_normalized.mp3"

    def operation(
        paths: list[Path],
        output: Path,
        durations: list[float],
        callback: ProgressCallback | None,
    ) -> None:
        _run_ffmpeg(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(paths[0]),
                "-map",
                "0:a:0",
                "-vn",
                "-af",
                f"loudnorm=I={target_lufs}:TP=-1.5:LRA=11",
                "-c:a",
                "libmp3lame",
                "-q:a",
                "2",
                str(output),
            ],
            durations[0],
            callback,
        )

    return _process_audio(
        [(input_data, input_name)], output_name, "audio/mpeg", operation, progress
    )


def merge_audio(
    inputs: list[tuple[bytes, str]],
    output_format: str,
    progress: ProgressCallback | None = None,
) -> VideoResult:
    """Concatenate two or more audio files and encode them into one file."""
    if len(inputs) < 2:
        raise ValueError("Select at least two audio files to merge.")
    output_format = output_format.lower().lstrip(".")
    if output_format not in {"mp3", "wav", "m4a"}:
        raise ValueError("Choose MP3, WAV, or M4A as the merged output format.")
    codec, mime_type = _FORMATS[output_format]
    first_stem = Path(inputs[0][1]).stem
    output_name = f"{first_stem}_merged.{output_format}"

    def operation(
        paths: list[Path],
        output: Path,
        durations: list[float],
        callback: ProgressCallback | None,
    ) -> None:
        command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y"]
        for path in paths:
            command.extend(["-i", str(path)])
        streams = "".join(f"[{index}:a:0]" for index in range(len(paths)))
        command.extend(
            [
                "-filter_complex",
                f"{streams}concat=n={len(paths)}:v=0:a=1[outa]",
                "-map",
                "[outa]",
                "-c:a",
                codec,
            ]
        )
        if output_format == "m4a":
            command.extend(["-movflags", "+faststart"])
        command.append(str(output))
        _run_ffmpeg(command, sum(durations), callback)

    return _process_audio(inputs, output_name, mime_type, operation, progress)
