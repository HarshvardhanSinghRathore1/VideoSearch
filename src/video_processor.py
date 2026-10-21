"""
Video Processor Module
Handles video validation, format verification, audio extraction using FFmpeg,
and audio/video metadata extraction without modifying original files.
"""

import os
import shutil
import subprocess
import logging
from pathlib import Path
from typing import Dict, Any, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("VideoProcessor")

SUPPORTED_FORMATS = {".mp4", ".mkv", ".mov", ".avi", ".webm"}


class VideoProcessor:
    """Extracts audio and metadata from video files using FFmpeg."""

    def __init__(self, output_audio_dir: str = "data/audio"):
        self.output_audio_dir = Path(output_audio_dir)
        self.output_audio_dir.mkdir(parents=True, exist_ok=True)
        self.ffmpeg_exe = self._resolve_ffmpeg()

    def _resolve_ffmpeg(self) -> str:
        """Finds system ffmpeg or falls back to imageio-ffmpeg bundled binary."""
        # Check system PATH first
        system_ffmpeg = shutil.which("ffmpeg")
        if system_ffmpeg:
            logger.info(f"Using system FFmpeg: {system_ffmpeg}")
            return system_ffmpeg

        # Fallback to imageio-ffmpeg
        try:
            import imageio_ffmpeg
            ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
            logger.info(f"Using imageio-ffmpeg binary: {ffmpeg_exe}")
            return ffmpeg_exe
        except ImportError:
            raise RuntimeError(
                "FFmpeg not found in system PATH and imageio-ffmpeg is not installed. "
                "Please install FFmpeg or run `pip install imageio-ffmpeg`."
            )

    def validate_video_path(self, video_path: str | Path) -> Path:
        """Validates video existence and format."""
        path = Path(video_path)
        if not path.exists():
            raise FileNotFoundError(f"Video file not found at: {path.resolve()}")
        if not path.is_file():
            raise ValueError(f"Path provided is not a file: {path.resolve()}")
        if path.suffix.lower() not in SUPPORTED_FORMATS:
            raise ValueError(
                f"Unsupported video format: '{path.suffix}'. "
                f"Supported formats are: {', '.join(sorted(SUPPORTED_FORMATS))}"
            )
        return path

    def extract_audio(
        self,
        video_path: str | Path,
        output_format: str = "wav",
        sample_rate: int = 16000,
        channels: int = 1,
        overwrite: bool = True,
    ) -> Path:
        """
        Extracts an audio stream from the given video file.

        Args:
            video_path: Path to the input video file.
            output_format: Target audio format (default 'wav').
            sample_rate: Target audio sample rate in Hz (default 16000 for Whisper).
            channels: Target audio channels (default 1 for mono).
            overwrite: Whether to overwrite existing extracted audio.

        Returns:
            Path to the extracted audio file.
        """
        v_path = self.validate_video_path(video_path)
        output_filename = f"{v_path.stem}.{output_format}"
        output_path = self.output_audio_dir / output_filename

        if output_path.exists() and not overwrite:
            logger.info(f"Audio file already exists at {output_path}. Skipping extraction.")
            return output_path

        cmd = [
            self.ffmpeg_exe,
            "-y" if overwrite else "-n",
            "-i", str(v_path.resolve()),
            "-vn",                       # Disable video stream
            "-acodec", "pcm_s16le" if output_format == "wav" else "copy",
            "-ar", str(sample_rate),     # Audio sampling rate
            "-ac", str(channels),        # Audio channels
            str(output_path.resolve()),
        ]

        logger.info(f"Extracting audio from {v_path.name} to {output_path.name}...")
        try:
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
            logger.info(f"Successfully extracted audio to {output_path}")
            return output_path
        except subprocess.CalledProcessError as e:
            logger.error(f"FFmpeg audio extraction failed: {e.stderr}")
            raise RuntimeError(f"FFmpeg failed with return code {e.returncode}: {e.stderr}")

    def get_media_metadata(self, file_path: str | Path) -> Dict[str, Any]:
        """Extracts basic metadata about the video/audio file."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        file_size_bytes = path.stat().st_size
        file_size_mb = round(file_size_bytes / (1024 * 1024), 2)

        return {
            "filename": path.name,
            "filepath": str(path.resolve()),
            "extension": path.suffix.lower(),
            "size_mb": file_size_mb,
            "is_video": path.suffix.lower() in SUPPORTED_FORMATS,
        }
