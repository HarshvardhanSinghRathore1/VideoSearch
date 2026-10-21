"""
Transcript Chunking Module
Implements timestamp-preserving semantic and windowed chunking over timestamped transcript segments.
Ensures every chunk maintains exact temporal boundaries (start_time, end_time) and source video reference.
"""

import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Any, Optional
from src.transcription import TranscriptSegment, format_timestamp

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("TranscriptChunker")


@dataclass
class TranscriptChunk:
    """A timestamp-bounded chunk composed of aggregated transcript segments."""
    chunk_id: str
    text: str
    start_time: float
    end_time: float
    source_video: str
    token_count_approx: int = 0
    segment_indices: Optional[List[int]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @property
    def duration(self) -> float:
        return max(0.0, self.end_time - self.start_time)

    def formatted_time_range(self) -> str:
        return f"[{format_timestamp(self.start_time)} - {format_timestamp(self.end_time)}]"


class TranscriptChunker:
    """Chunks timestamped transcript segments with configurable window size and overlap."""

    def __init__(
        self,
        target_chunk_size: int = 120,   # approximate token / word limit (100-300 tokens)
        chunk_overlap: int = 25,        # approximate token / word overlap
        min_chunk_size: int = 20,       # minimum tokens to form a standalone chunk
    ):
        if chunk_overlap >= target_chunk_size:
            raise ValueError(f"chunk_overlap ({chunk_overlap}) must be strictly less than target_chunk_size ({target_chunk_size})")

        self.target_chunk_size = target_chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size

    def _estimate_tokens(self, text: str) -> int:
        """Heuristic token count: ~1.3 tokens per whitespace-separated word."""
        words = text.strip().split()
        return max(1, int(len(words) * 1.3))

    def chunk_segments(
        self,
        segments: List[TranscriptSegment],
        source_video: str = "sample.mp4",
    ) -> List[TranscriptChunk]:
        """
        Aggregates consecutive transcript segments into timestamped chunks.

        Maintains exact start_time from the first segment and end_time from the last segment in each chunk.
        Slides forward by subtracting overlap from accumulated tokens.

        Args:
            segments: List of TranscriptSegment objects.
            source_video: Name or identifier of the source video.

        Returns:
            List of TranscriptChunk objects.
        """
        if not segments:
            logger.warning("Empty segments provided to chunker.")
            return []

        chunks: List[TranscriptChunk] = []
        seg_count = len(segments)
        i = 0
        chunk_idx = 1

        while i < seg_count:
            curr_segments: List[TranscriptSegment] = []
            curr_indices: List[int] = []
            curr_tokens = 0
            j = i

            while j < seg_count:
                seg = segments[j]
                seg_tokens = self._estimate_tokens(seg.text)

                if curr_segments and (curr_tokens + seg_tokens > self.target_chunk_size):
                    break

                curr_segments.append(seg)
                curr_indices.append(j)
                curr_tokens += seg_tokens
                j += 1

            if not curr_segments:
                curr_segments.append(segments[i])
                curr_indices.append(i)
                j = i + 1

            combined_text = " ".join(s.text.strip() for s in curr_segments)
            start_t = round(curr_segments[0].start, 2)
            end_t = round(curr_segments[-1].end, 2)

            chunk = TranscriptChunk(
                chunk_id=f"chunk_{chunk_idx:03d}",
                text=combined_text,
                start_time=start_t,
                end_time=end_t,
                source_video=source_video,
                token_count_approx=self._estimate_tokens(combined_text),
                segment_indices=curr_indices,
            )
            chunks.append(chunk)
            chunk_idx += 1

            if j >= seg_count:
                break

            # Calculate overlap advancement
            # Backtrack segments to maintain approximately `chunk_overlap` tokens
            overlap_tokens = 0
            overlap_start_idx = j
            for k in reversed(curr_indices):
                seg_k_tokens = self._estimate_tokens(segments[k].text)
                if overlap_tokens + seg_k_tokens <= self.chunk_overlap:
                    overlap_tokens += seg_k_tokens
                    overlap_start_idx = k
                else:
                    break

            # Ensure we always make forward progress
            if overlap_start_idx <= i:
                overlap_start_idx = i + 1

            i = overlap_start_idx

        logger.info(
            f"Chunking complete for {source_video}: Generated {len(chunks)} chunks "
            f"from {len(segments)} segments (target_size={self.target_chunk_size}, overlap={self.chunk_overlap})."
        )
        return chunks

    def save_chunks(self, chunks: List[TranscriptChunk], output_path: str | Path) -> Path:
        """Saves chunks to a JSON file."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = [chunk.to_dict() for chunk in chunks]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        logger.info(f"Saved {len(chunks)} chunks to {path}")
        return path

    @staticmethod
    def load_chunks(json_path: str | Path) -> List[TranscriptChunk]:
        """Loads chunks from a JSON file."""
        path = Path(json_path)
        if not path.exists():
            raise FileNotFoundError(f"Chunks file not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        return [
            TranscriptChunk(
                chunk_id=item["chunk_id"],
                text=item["text"],
                start_time=item["start_time"],
                end_time=item["end_time"],
                source_video=item["source_video"],
                token_count_approx=item.get("token_count_approx", 0),
                segment_indices=item.get("segment_indices"),
            )
            for item in data
        ]
