"""FFmpeg-backed video operations used by the Streamlit tool pages."""

import json
import os
import subprocess
import threading
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path


class VideoProcessingError(Exception):
    """An actionable failure while inspecting or processing a video."""


@dataclass(frozen=True)
class VideoInfo:
    duration: float
    has_audio: bool


@dataclass(frozen=True)
class VideoResult:
    data: bytes
    filename: str
    mime_type: str


ProgressCallback = Callable[[float], None]


def probe_video(input_path: Path) -> VideoInfo:
    """Read duration and audio presence from a video using ffprobe."""
    command = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration:stream=codec_type",
        "-of",
        "json",
        str(input_path),
    ]
    try:
        result = subprocess.run(
            command,
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
        raise VideoProcessingError("Video inspection timed out.") from exc
    except subprocess.CalledProcessError as exc:
        detail = exc.stderr.strip() or "Unsupported or invalid video file."
        raise VideoProcessingError(f"Could not read video metadata: {detail}") from exc

    try:
        metadata = json.loads(result.stdout)
        duration = float(metadata["format"]["duration"])
        streams = metadata.get("streams", [])
        has_video = any(stream.get("codec_type") == "video" for stream in streams)
        has_audio = any(stream.get("codec_type") == "audio" for stream in streams)
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise VideoProcessingError(
            "Could not determine the video duration. Try another video file."
        ) from exc

    if duration <= 0:
        raise VideoProcessingError("The uploaded video has no usable duration.")
    if not has_video:
        raise VideoProcessingError("The uploaded file does not contain a video stream.")
    return VideoInfo(duration=duration, has_audio=has_audio)


def _run_ffmpeg(
    command: list[str],
    duration: float,
    progress: ProgressCallback | None = None,
    progress_start: float = 0.0,
    progress_end: float = 1.0,
    timeout: int = 900,
) -> None:
    command = [
        *command[:-1],
        "-progress",
        "pipe:1",
        "-nostats",
        command[-1],
    ]
    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
    except FileNotFoundError as exc:
        raise VideoProcessingError(
            "ffmpeg was not found. Install FFmpeg and add its bin folder to PATH."
        ) from exc

    errors: list[str] = []

    def read_progress() -> None:
        if process.stdout is None:
            return
        current_time = 0.0
        for raw_line in process.stdout:
            line = raw_line.strip()
            key, separator, value = line.partition("=")
            if not separator:
                if line and len(errors) < 50:
                    errors.append(line)
                continue
            if key == "out_time_ms":
                try:
                    current_time = int(value) / 1_000_000
                except ValueError:
                    continue
                if progress is not None and duration > 0:
                    fraction = min(1.0, current_time / duration)
                    progress(
                        progress_start + fraction * (progress_end - progress_start)
                    )
            elif key == "progress" and value == "end" and progress is not None:
                progress(progress_end)

    reader = threading.Thread(target=read_progress, daemon=True)
    reader.start()
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        process.kill()
        process.wait()
        raise VideoProcessingError(
            "Media processing timed out. Try a shorter or smaller input file."
        ) from exc
    finally:
        reader.join()

    if process.returncode:
        detail = "\n".join(errors).strip() or "FFmpeg could not process this file."
        raise VideoProcessingError(f"Media processing failed: {detail}")


def _process(
    input_data: bytes,
    input_name: str,
    output_name: str,
    mime_type: str,
    command_builder: Callable[
        [Path, Path, Path, ProgressCallback | None], None
    ],
    progress: ProgressCallback | None = None,
) -> VideoResult:
    suffix = Path(input_name).suffix or ".mp4"
    with tempfile.TemporaryDirectory(prefix="everyday-tools-video-") as temp_dir:
        work_dir = Path(temp_dir)
        input_path = work_dir / f"input{suffix}"
        output_path = work_dir / output_name
        input_path.write_bytes(input_data)
        command_builder(input_path, output_path, work_dir, progress)
        if not output_path.is_file() or output_path.stat().st_size == 0:
            raise VideoProcessingError(
                "FFmpeg finished without creating an output file."
            )
        output_data = output_path.read_bytes()

    return VideoResult(
        data=output_data,
        filename=output_name,
        mime_type=mime_type,
    )


def compress_video(
    input_data: bytes,
    input_name: str,
    reduction_percent: int,
    progress: ProgressCallback | None = None,
) -> VideoResult:
    """Compress a video toward a requested percentage size reduction."""
    if not 1 <= reduction_percent <= 90:
        raise ValueError("Choose a size reduction between 1% and 90%.")
    output_name = f"{Path(input_name).stem}_compressed.mp4"

    def build(
        input_path: Path,
        output_path: Path,
        work_dir: Path,
        progress_callback: ProgressCallback | None,
    ) -> None:
        info = probe_video(input_path)
        target_bytes = input_path.stat().st_size * (
            1 - reduction_percent / 100
        )
        total_bitrate = target_bytes * 8 / info.duration
        audio_bitrate = 128_000 if info.has_audio else 0
        video_bitrate = max(64_000, int(total_bitrate - audio_bitrate))
        passlog = str(work_dir / "ffmpeg-pass")
        null_device = "NUL" if os.name == "nt" else os.devnull
        common = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(input_path),
            "-map",
            "0:v:0",
            "-map",
            "0:a?",
            "-c:v",
            "libx264",
            "-b:v",
            str(video_bitrate),
            "-passlogfile",
            passlog,
        ]
        pass_one = [*common, "-pass", "1", "-an", "-f", "null", null_device]
        _run_ffmpeg(
            pass_one,
            info.duration,
            progress_callback,
            progress_start=0.0,
            progress_end=0.5,
        )

        pass_two = [
            *common,
            "-pass",
            "2",
            "-movflags",
            "+faststart",
        ]
        if info.has_audio:
            pass_two.extend(["-c:a", "aac", "-b:a", str(audio_bitrate)])
        else:
            pass_two.append("-an")
        pass_two.append(str(output_path))
        _run_ffmpeg(
            pass_two,
            info.duration,
            progress_callback,
            progress_start=0.5,
            progress_end=1.0,
        )

    return _process(
        input_data, input_name, output_name, "video/mp4", build, progress
    )


def convert_video(
    input_data: bytes,
    input_name: str,
    output_format: str,
    progress: ProgressCallback | None = None,
) -> VideoResult:
    """Convert a video into a supported container and codec combination."""
    codecs = {
        "mp4": ("libx264", "aac"),
        "mkv": ("libx264", "aac"),
        "mov": ("libx264", "aac"),
        "webm": ("libvpx-vp9", "libopus"),
    }
    output_format = output_format.lower().lstrip(".")
    if output_format not in codecs:
        raise ValueError("Choose MP4, MKV, MOV, or WebM as the output format.")
    video_codec, audio_codec = codecs[output_format]
    mime_types = {
        "mp4": "video/mp4",
        "mkv": "video/x-matroska",
        "mov": "video/quicktime",
        "webm": "video/webm",
    }
    output_name = f"{Path(input_name).stem}_converted.{output_format}"

    def build(
        input_path: Path,
        output_path: Path,
        _work_dir: Path,
        progress_callback: ProgressCallback | None,
    ) -> None:
        info = probe_video(input_path)
        command = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(input_path),
            "-map",
            "0:v:0",
            "-map",
            "0:a?",
            "-c:v",
            video_codec,
            "-c:a",
            audio_codec,
        ]
        if output_format in {"mp4", "mov"}:
            command.extend(["-movflags", "+faststart"])
        command.append(str(output_path))
        _run_ffmpeg(command, info.duration, progress_callback)

    return _process(
        input_data,
        input_name,
        output_name,
        mime_types[output_format],
        build,
        progress,
    )


def trim_video(
    input_data: bytes,
    input_name: str,
    start_seconds: float,
    end_seconds: float,
    progress: ProgressCallback | None = None,
) -> VideoResult:
    """Trim a video between the requested start and end times."""
    if start_seconds < 0 or end_seconds <= start_seconds:
        raise ValueError("End time must be greater than a non-negative start time.")
    output_name = f"{Path(input_name).stem}_trimmed.mp4"

    def build(
        input_path: Path,
        output_path: Path,
        _work_dir: Path,
        progress_callback: ProgressCallback | None,
    ) -> None:
        info = probe_video(input_path)
        if end_seconds > info.duration:
            raise ValueError("End time cannot be after the end of the video.")
        duration = end_seconds - start_seconds
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
                str(input_path),
                "-t",
                str(duration),
                "-map",
                "0:v:0",
                "-map",
                "0:a?",
                "-c:v",
                "libx264",
                "-c:a",
                "aac",
                "-movflags",
                "+faststart",
                str(output_path),
            ],
            duration,
            progress_callback,
        )

    return _process(
        input_data, input_name, output_name, "video/mp4", build, progress
    )


def make_gif(
    input_data: bytes,
    input_name: str,
    start_seconds: float,
    duration_seconds: float,
    fps: int,
    width: int,
    progress: ProgressCallback | None = None,
) -> VideoResult:
    """Create an optimized palette-based GIF from a video segment."""
    if start_seconds < 0 or not 1 <= duration_seconds <= 30:
        raise ValueError("Choose a start time and a GIF duration from 1 to 30 seconds.")
    if not 5 <= fps <= 20 or not 160 <= width <= 960:
        raise ValueError("Choose a frame rate from 5–20 fps and width from 160–960 px.")
    output_name = f"{Path(input_name).stem}.gif"

    def build(
        input_path: Path,
        output_path: Path,
        _work_dir: Path,
        progress_callback: ProgressCallback | None,
    ) -> None:
        info = probe_video(input_path)
        if start_seconds + duration_seconds > info.duration:
            raise ValueError("GIF segment must end before the video ends.")
        _run_ffmpeg(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-ss",
                str(start_seconds),
                "-t",
                str(duration_seconds),
                "-i",
                str(input_path),
                "-filter_complex",
                (
                    f"fps={fps},scale={width}:-1:flags=lanczos,"
                    "split[a][b];[a]palettegen[p];[b][p]paletteuse"
                ),
                str(output_path),
            ],
            duration_seconds,
            progress_callback,
        )

    return _process(
        input_data, input_name, output_name, "image/gif", build, progress
    )


def extract_thumbnail(
    input_data: bytes,
    input_name: str,
    timestamp_seconds: float,
    image_format: str,
    quality: int,
    progress: ProgressCallback | None = None,
) -> VideoResult:
    """Extract a frame at a given time as a JPG or PNG image."""
    image_format = image_format.lower()
    if image_format not in {"jpg", "png"}:
        raise ValueError("Choose JPG or PNG as the thumbnail format.")
    if not 1 <= quality <= 100:
        raise ValueError("Choose an image quality between 1 and 100.")
    output_name = f"{Path(input_name).stem}_thumbnail.{image_format}"

    def build(
        input_path: Path,
        output_path: Path,
        _work_dir: Path,
        progress_callback: ProgressCallback | None,
    ) -> None:
        info = probe_video(input_path)
        if timestamp_seconds >= info.duration:
            raise ValueError("Frame time must be before the end of the video.")
        command = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-ss",
            str(timestamp_seconds),
            "-i",
            str(input_path),
            "-frames:v",
            "1",
        ]
        if image_format == "jpg":
            command.extend(["-q:v", str(max(2, round((100 - quality) / 4) + 2))])
        command.append(str(output_path))
        _run_ffmpeg(command, info.duration, progress_callback)

    mime_type = "image/jpeg" if image_format == "jpg" else "image/png"
    return _process(
        input_data, input_name, output_name, mime_type, build, progress
    )
