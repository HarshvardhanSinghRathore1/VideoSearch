"""
Transcription Module
Performs speech-to-text with timestamp preservation using faster-whisper or openai-whisper.
Outputs structured JSON and human-readable formatted TXT transcripts.
"""

import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Any, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("Transcriber")


@dataclass
class TranscriptSegment:
    """Individual timestamped speech segment."""
    start: float
    end: float
    text: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


def format_timestamp(seconds: float) -> str:
    """Converts seconds into HH:MM:SS.mmm format."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:05.2f}"


class Transcriber:
    """Speech-to-text transcriber with timestamp alignment."""

    def __init__(
        self,
        model_size: str = "base",
        device: str = "cpu",
        compute_type: str = "int8",
        output_dir: str = "data/transcripts",
    ):
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._model = None
        self._backend = None

    def _load_model(self):
        """Lazy loads faster-whisper or standard whisper."""
        if self._model is not None:
            return

        try:
            from faster_whisper import WhisperModel
            logger.info(f"Loading faster-whisper model '{self.model_size}' on {self.device} ({self.compute_type})...")
            self._model = WhisperModel(self.model_size, device=self.device, compute_type=self.compute_type)
            self._backend = "faster-whisper"
            logger.info("faster-whisper loaded successfully.")
            return
        except ImportError:
            logger.warning("faster-whisper not available. Trying standard whisper...")

        try:
            import whisper
            logger.info(f"Loading openai-whisper model '{self.model_size}' on {self.device}...")
            self._model = whisper.load_model(self.model_size, device=self.device)
            self._backend = "openai-whisper"
            logger.info("openai-whisper loaded successfully.")
            return
        except ImportError:
            raise ImportError(
                "Neither faster-whisper nor openai-whisper is installed. "
                "Please install with `pip install faster-whisper` or `pip install openai-whisper`."
            )

    def transcribe(
        self,
        audio_path: str | Path,
        language: Optional[str] = "en",
        beam_size: int = 5,
        word_timestamps: bool = False,
    ) -> List[TranscriptSegment]:
        """
        Transcribes an audio file into timestamped segments.

        Args:
            audio_path: Path to audio file.
            language: Spoken language code (e.g., 'en').
            beam_size: Beam search width.
            word_timestamps: Whether to extract word-level timestamps.

        Returns:
            List of TranscriptSegment instances.
        """
        self._load_model()
        path = Path(audio_path)
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {path.resolve()}")

        logger.info(f"Transcribing {path.name} using backend: {self._backend}...")
        segments: List[TranscriptSegment] = []

        if self._backend == "faster-whisper":
            fw_segments, info = self._model.transcribe(
                str(path.resolve()),
                language=language,
                beam_size=beam_size,
                word_timestamps=word_timestamps,
            )
            logger.info(f"Detected language '{info.language}' with probability {info.language_probability:.2f}")
            for seg in fw_segments:
                clean_text = seg.text.strip()
                if clean_text:
                    segments.append(TranscriptSegment(
                        start=round(seg.start, 2),
                        end=round(seg.end, 2),
                        text=clean_text
                    ))
        else:
            # openai-whisper backend
            result = self._model.transcribe(
                str(path.resolve()),
                language=language,
                beam_size=beam_size,
            )
            for seg in result.get("segments", []):
                clean_text = seg.get("text", "").strip()
                if clean_text:
                    segments.append(TranscriptSegment(
                        start=round(float(seg["start"]), 2),
                        end=round(float(seg["end"]), 2),
                        text=clean_text
                    ))

        logger.info(f"Transcription completed. Extracted {len(segments)} timestamped segments.")
        return segments

    def save_transcripts(
        self,
        segments: List[TranscriptSegment],
        video_name: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Path]:
        """
        Saves transcript in both structured JSON and formatted human-readable TXT formats.

        Args:
            segments: List of timestamped segments.
            video_name: Identifier or filename of the source video.
            metadata: Additional metadata dictionary.

        Returns:
            Dictionary with paths to the saved 'json' and 'txt' files.
        """
        base_stem = Path(video_name).stem
        json_path = self.output_dir / f"{base_stem}_transcript.json"
        txt_path = self.output_dir / f"{base_stem}_transcript.txt"

        # 1. Save Structured JSON
        data = {
            "source_video": video_name,
            "segment_count": len(segments),
            "total_duration_sec": round(segments[-1].end if segments else 0.0, 2),
            "metadata": metadata or {},
            "segments": [seg.to_dict() for seg in segments],
        }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        # 2. Save Human-Readable TXT
        lines = [
            f"# Video Transcript: {video_name}",
            f"# Segments: {len(segments)} | Total Duration: {format_timestamp(data['total_duration_sec'])}",
            "=" * 70,
            ""
        ]
        for seg in segments:
            t_start = format_timestamp(seg.start)
            t_end = format_timestamp(seg.end)
            lines.append(f"[{t_start} - {t_end}]")
            lines.append(f"{seg.text}\n")

        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        logger.info(f"Saved transcripts to:\n  JSON: {json_path}\n  TXT:  {txt_path}")
        return {"json": json_path, "txt": txt_path}

    @staticmethod
    def load_transcript_json(json_path: str | Path) -> List[TranscriptSegment]:
        """Loads transcript segments from a saved JSON file."""
        path = Path(json_path)
        if not path.exists():
            raise FileNotFoundError(f"Transcript file not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        segments = [
            TranscriptSegment(
                start=item["start"],
                end=item["end"],
                text=item["text"]
            )
            for item in data.get("segments", [])
        ]
        return segments
