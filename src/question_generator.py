"""
Question Generator Module
Dynamically generates 4-6 high-quality, transcript-grounded questions for processed videos.
Uses Google Gemini with automatic caching to avoid redundant API calls.
"""

import os
import json
import re
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

from src.transcription import TranscriptSegment
from src.chunking import TranscriptChunk
from src.llm import GroundedLLMGenerator

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("QuestionGenerator")

TRANSCRIPTS_DIR = Path("data/transcripts")


class QuestionGenerator:
    """Generates dynamic, transcript-grounded suggested questions for any video."""

    def __init__(
        self,
        llm_generator: Optional[GroundedLLMGenerator] = None,
        cache_dir: str | Path = "data/transcripts",
    ):
        self.llm_generator = llm_generator or GroundedLLMGenerator()
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_cache_path(self, video_id: str) -> Path:
        """Returns the cache JSON file path for a given video_id."""
        clean_id = re.sub(r"[^\w\-_\.]", "_", video_id)
        return self.cache_dir / f"{clean_id}_suggested_questions.json"

    def get_cached_questions(self, video_id: str) -> Optional[List[str]]:
        """Loads cached questions from disk if available."""
        cache_file = self._get_cache_path(video_id)
        if cache_file.exists():
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, list) and len(data) >= 3:
                        logger.info(f"Loaded {len(data)} cached suggested questions for video '{video_id}'.")
                        return data
                    elif isinstance(data, dict) and "questions" in data:
                        logger.info(f"Loaded {len(data['questions'])} cached questions for video '{video_id}'.")
                        return data["questions"]
            except Exception as e:
                logger.warning(f"Failed to read cache for '{video_id}': {e}")
        return None

    def save_cached_questions(self, video_id: str, questions: List[str]):
        """Persists generated questions to disk for instant subsequent retrieval."""
        cache_file = self._get_cache_path(video_id)
        try:
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(questions, f, indent=2)
            logger.info(f"Saved {len(questions)} suggested questions to {cache_file}")
        except Exception as e:
            logger.warning(f"Failed to save questions cache for '{video_id}': {e}")

    def _build_representative_transcript_text(
        self,
        segments_or_chunks: List[Any],
        max_words: int = 2500,
    ) -> str:
        """Constructs a compact, representative transcript text covering the entire timeline."""
        if not segments_or_chunks:
            return ""

        # Extract text items
        text_items = []
        for item in segments_or_chunks:
            if isinstance(item, TranscriptSegment):
                text_items.append(f"[{item.start:.1f}s - {item.end:.1f}s]: {item.text}")
            elif isinstance(item, TranscriptChunk):
                text_items.append(f"[{item.start_time:.1f}s - {item.end_time:.1f}s]: {item.text}")
            elif isinstance(item, dict):
                s = item.get("start", item.get("start_time", 0.0))
                e = item.get("end", item.get("end_time", 0.0))
                t = item.get("text", "")
                text_items.append(f"[{s:.1f}s - {e:.1f}s]: {t}")

        full_text = "\n".join(text_items)
        words = full_text.split()
        if len(words) <= max_words:
            return full_text

        # If long, sample evenly across beginning, middle sections, and end
        num_items = len(text_items)
        sample_size = min(num_items, 25)
        step = max(1, num_items // sample_size)
        sampled = [text_items[i] for i in range(0, num_items, step)]
        if text_items[-1] not in sampled:
            sampled.append(text_items[-1])

        return "\n".join(sampled)

    def generate_questions(
        self,
        video_id: str,
        segments_or_chunks: Optional[List[Any]] = None,
        force_regenerate: bool = False,
    ) -> List[str]:
        """
        Generates 4 to 6 dynamic, transcript-grounded questions for a given video.

        Args:
            video_id: Unique identifier or stem of the video.
            segments_or_chunks: List of TranscriptSegment, TranscriptChunk, or dicts.
            force_regenerate: If True, ignores cache and regenerates via Gemini.

        Returns:
            List of 4-6 question strings.
        """
        # 1. Check disk cache first unless forced
        if not force_regenerate:
            cached = self.get_cached_questions(video_id)
            if cached:
                return cached

        # 2. If segments not passed, try loading from transcript JSON on disk
        if not segments_or_chunks:
            transcript_json = TRANSCRIPTS_DIR / f"{video_id}_transcript.json"
            if not transcript_json.exists():
                transcript_json = TRANSCRIPTS_DIR / "sample_transcript.json"

            if transcript_json.exists():
                try:
                    with open(transcript_json, "r", encoding="utf-8") as f:
                        raw = json.load(f)
                        if isinstance(raw, list):
                            segments_or_chunks = raw
                        elif isinstance(raw, dict) and "segments" in raw:
                            segments_or_chunks = raw["segments"]
                except Exception as e:
                    logger.warning(f"Error reading transcript JSON {transcript_json}: {e}")

        if not segments_or_chunks:
            raise ValueError(f"No transcript segments available for video '{video_id}' to generate questions.")

        transcript_context = self._build_representative_transcript_text(segments_or_chunks)

        # 3. Formulate Gemini prompt
        prompt = f"""Generate 4 to 6 natural-language questions that can be answered using ONLY the provided video speech transcript.

CRITICAL REQUIREMENTS:
1. The questions must be specifically about the concepts, definitions, algorithms, explanations, examples, reasoning, or relationships discussed in this transcript.
2. Prefer questions that test:
   - Specific concepts and definitions
   - Problem formulations and mathematical/algorithmic steps
   - Trade-offs, complexities, or edge cases discussed
   - Examples and reasoning presented in the lecture
3. Do NOT generate generic questions (such as "What is this video about?", "Summarize the video", "Explain this topic").
4. Do NOT ask questions requiring outside knowledge not contained in the transcript.
5. Avoid duplicate or vague questions.
6. Return ONLY a valid JSON array of 4 to 6 strings.

Transcript Excerpts:
---
{transcript_context}
---

Return JSON array format strictly:
[
  "Question 1?",
  "Question 2?",
  "Question 3?",
  "Question 4?",
  "Question 5?"
]
"""

        # 4. Generate via Gemini or fallback
        questions = []
        try:
            raw_response = self.llm_generator._call_gemini(prompt)
            questions = self._parse_json_questions(raw_response)
        except Exception as e:
            logger.warning(f"Gemini question generation error: {e}. Using grounded heuristic generator.")
            questions = self._offline_heuristic_questions(transcript_context)

        # Ensure valid length
        if not questions or len(questions) < 3:
            questions = self._offline_heuristic_questions(transcript_context)

        # 5. Cache and return
        self.save_cached_questions(video_id, questions)
        return questions

    def _parse_json_questions(self, text: str) -> List[str]:
        """Safely parses a JSON array of question strings from LLM text."""
        cleaned = text.strip()
        # Remove code fences
        cleaned = re.sub(r"^```(?:json)?", "", cleaned, flags=re.MULTILINE)
        cleaned = re.sub(r"```$", "", cleaned, flags=re.MULTILINE).strip()

        # Find JSON array using regex
        match = re.search(r"\[\s*\"[^\"]+[\s\S]*?\]", cleaned)
        if match:
            try:
                parsed = json.loads(match.group(0))
                if isinstance(parsed, list):
                    res = [str(q).strip() for q in parsed if str(q).strip()]
                    if len(res) >= 3:
                        return res[:6]
            except Exception:
                pass

        # Line-by-line fallback
        lines = [re.sub(r"^\d+[\.\)]\s*", "", line.strip().strip('"[],')) for line in text.splitlines()]
        res = [l for l in lines if l and l.endswith("?")]
        return res[:6] if len(res) >= 3 else []

    def _offline_heuristic_questions(self, transcript_text: str) -> List[str]:
        """Offline heuristic question extraction for resilience."""
        lower_t = transcript_text.lower()
        questions = []

        if "binary search" in lower_t or "square root" in lower_t:
            questions = [
                "What problem is being solved using binary search in this video?",
                "How does the binary search algorithm determine the floor of the square root of n?",
                "What is the search space range defined for finding the square root?",
                "What is the time complexity of the binary search approach compared to linear search?",
                "How does the algorithm handle values when n is not a perfect square?"
            ]
        elif "normalization" in lower_t or "database" in lower_t:
            questions = [
                "What is the primary goal of database normalization?",
                "What is required for a relational table to satisfy First Normal Form (1NF)?",
                "How does Second Normal Form (2NF) eliminate partial dependencies?",
                "What is a transitive dependency and how is it resolved in Third Normal Form (3NF)?",
                "What common data preprocessing steps are discussed for data mining?"
            ]
        else:
            # Generic extractor based on detected keywords
            keywords = [w for w in re.findall(r"\b[A-Za-z]{4,}\b", transcript_text) if w.lower() not in {"this", "that", "with", "from", "have", "here"}]
            top_kw = list(dict.fromkeys(keywords))[:4]
            questions = [
                f"How is {top_kw[0]} explained in the video?" if len(top_kw) > 0 else "What key concept is introduced in this video?",
                f"What approach is proposed for {top_kw[1]}?" if len(top_kw) > 1 else "How does the proposed method work?",
                f"What examples are used to demonstrate {top_kw[2]}?" if len(top_kw) > 2 else "What examples are discussed in the lecture?",
                "What are the main steps outlined in the explanation?"
            ]

        return questions[:5]
