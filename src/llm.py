"""
LLM Generation Module
Interfaces with Google Gemini API (or offline fallback) to generate strict,
evidence-grounded answers citing video timestamps.
"""

import os
import time
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("GroundedLLMGenerator")

SYSTEM_PROMPT = """You are an AI research assistant answering questions about a video based STRICTLY on its timestamped speech transcript.

CRITICAL RULES:
1. Answer using ONLY the supplied transcript evidence below.
2. Do NOT invent or extrapolate facts not present in the context.
3. If the answer cannot be determined from the provided transcript excerpts, state explicitly: "The provided video transcript does not contain information to answer this question."
4. Always cite the specific evidence time range(s) (e.g. [00:01:23 - 00:01:45]) used to derive your answer.
5. Keep your answer clear, factual, and direct.

FORMAT YOUR RESPONSE EXACTLY AS FOLLOWS:
Answer: <Your concise, factual answer here>
Evidence: <Comma-separated timestamp ranges supporting the answer, e.g. [00:01:20 - 00:01:45], or 'None' if not found>
"""


class GroundedLLMGenerator:
    """Generates timestamp-grounded answers using Gemini or fallback generator."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "gemini-1.5-flash",
        temperature: float = 0.0,
    ):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.model_name = os.getenv("GEMINI_MODEL", model_name)
        self.temperature = temperature
        self._client = None
        self._sdk_type = None
        self._init_client()

    def _init_client(self):
        """Initializes Google GenAI client or google.generativeai."""
        if not self.api_key or self.api_key.strip() in ("", "your_gemini_api_key_here"):
            logger.warning(
                "GEMINI_API_KEY is not set. Generator will operate in offline/mock evidence-extraction mode. "
                "Set GEMINI_API_KEY in .env or pass api_key to use live Gemini LLM."
            )
            return

        try:
            # Try new google-genai SDK first
            from google import genai
            self._client = genai.Client(api_key=self.api_key)
            self._sdk_type = "google-genai"
            logger.info(f"Initialized Google GenAI client (model: {self.model_name}).")
            return
        except ImportError:
            pass

        try:
            # Fallback to legacy google-generativeai SDK
            import google.generativeai as genai_legacy
            genai_legacy.configure(api_key=self.api_key)
            self._client = genai_legacy.GenerativeModel(self.model_name)
            self._sdk_type = "google-generativeai"
            logger.info(f"Initialized google.generativeai client (model: {self.model_name}).")
            return
        except Exception as e:
            logger.warning(f"Failed to initialize Gemini SDK: {e}. Falling back to offline generator.")

    def generate_answer(
        self,
        query: str,
        retrieved_chunks: List[Dict[str, Any]],
        context_str: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Generates grounded answer and citation metadata from retrieved transcript chunks.

        Args:
            query: User's question.
            retrieved_chunks: List of chunk dictionaries from retriever.
            context_str: Pre-formatted context string (optional).

        Returns:
            Dictionary with 'answer', 'evidence_timestamps', 'retrieved_context', 'latency_sec', 'model'.
        """
        start_time = time.time()

        if context_str is None:
            from src.retriever import TranscriptRetriever
            # Reconstruct context block if not pre-provided
            dummy_retriever = TranscriptRetriever(None, None)
            context_str = dummy_retriever.format_context_for_llm(retrieved_chunks)

        user_prompt = (
            f"Question: {query}\n\n"
            f"Retrieved Video Transcript Context:\n{context_str}\n\n"
            f"Please answer the question based strictly on the excerpts above."
        )

        # Case 1: Live Gemini API Call
        if self._client is not None:
            try:
                raw_response = self._call_gemini(user_prompt)
                latency = round(time.time() - start_time, 3)
                parsed = self._parse_llm_response(raw_response, retrieved_chunks)
                parsed["latency_sec"] = latency
                parsed["model"] = f"gemini:{self.model_name}"
                parsed["retrieved_context"] = context_str
                return parsed
            except Exception as e:
                logger.error(f"Gemini generation error: {e}. Falling back to offline synthesis.")

        # Case 2: Offline / Mock Grounded Extractor
        latency = round(time.time() - start_time, 3)
        parsed = self._offline_grounded_answer(query, retrieved_chunks, context_str)
        parsed["latency_sec"] = latency
        parsed["model"] = "offline-grounded-baseline"
        parsed["retrieved_context"] = context_str
        return parsed

    def _call_gemini(self, prompt: str) -> str:
        """Invokes Gemini API via appropriate SDK with automatic fallback models."""
        candidate_models = [self.model_name, "gemini-3.5-flash", "gemini-3.5-flash-lite", "gemini-3.7-flash", "gemini-3.6-flash", "gemini-flash-latest"]
        # Filter duplicates preserving order
        candidate_models = list(dict.fromkeys([m for m in candidate_models if m]))
        last_err = None

        for model in candidate_models:
            if not model:
                continue
            try:
                if self._sdk_type == "google-genai":
                    response = self._client.models.generate_content(
                        model=model,
                        contents=prompt,
                        config={"system_instruction": SYSTEM_PROMPT, "temperature": self.temperature},
                    )
                    self.model_name = model
                    return response.text
                elif self._sdk_type == "google-generativeai":
                    import google.generativeai as genai_legacy
                    m = genai_legacy.GenerativeModel(model)
                    full_prompt = f"{SYSTEM_PROMPT}\n\n{prompt}"
                    response = m.generate_content(
                        full_prompt,
                        generation_config={"temperature": self.temperature},
                    )
                    self.model_name = model
                    return response.text
            except Exception as e:
                last_err = e
                logger.warning(f"Model '{model}' attempt error: {e}. Trying next candidate...")

        raise RuntimeError(f"All Gemini model attempts failed. Last error: {last_err}")

    def _parse_llm_response(self, raw_text: str, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Parses answer and evidence timestamps from model output."""
        lines = raw_text.strip().split("\n")
        answer_lines = []
        evidence_lines = []
        target = answer_lines

        for line in lines:
            if line.lower().startswith("answer:"):
                target = answer_lines
                answer_lines.append(line.split(":", 1)[1].strip())
            elif line.lower().startswith("evidence:"):
                target = evidence_lines
                evidence_lines.append(line.split(":", 1)[1].strip())
            else:
                target.append(line)

        answer_text = "\n".join(answer_lines).strip() or raw_text.strip()
        evidence_text = "\n".join(evidence_lines).strip()

        evidence_ts = []
        for c in chunks:
            evidence_ts.append({
                "chunk_id": c["chunk_id"],
                "start": c["start_time"],
                "end": c["end_time"],
                "formatted": c.get("formatted_time", f"[{c['start_time']} - {c['end_time']}]"),
            })

        return {
            "answer": answer_text,
            "evidence_raw": evidence_text,
            "evidence_timestamps": evidence_ts,
            "raw_response": raw_text,
        }

    def _offline_grounded_answer(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        context_str: str,
    ) -> Dict[str, Any]:
        """
        Offline fallback answering logic for deterministic testing when API key is not present.
        Inspects retrieval relevance and extracts direct context or signals refusal.
        """
        # Calibrated threshold for BGE-small-en cosine similarity
        refusal_response = {
            "answer": "The provided video transcript does not contain information to answer this question.",
            "evidence_raw": "None",
            "evidence_timestamps": [],
            "raw_response": "Answer: The provided video transcript does not contain information to answer this question.\nEvidence: None",
        }

        if not chunks:
            return refusal_response

        top_score = chunks[0].get("score", 0.0)
        # Check keyword presence or score threshold
        query_words = set(query.lower().replace("?", "").replace(".", "").split())
        stop_words = {"what", "is", "the", "how", "does", "in", "to", "and", "a", "an", "of", "for"}
        meaningful_words = query_words - stop_words

        has_overlap = any(w in chunks[0]["text"].lower() for w in meaningful_words)
        if top_score < 0.55 or not has_overlap:
            return refusal_response

        top_chunk = chunks[0]
        st = top_chunk["start_time"]
        et = top_chunk["end_time"]
        time_fmt = top_chunk.get("formatted_time", f"[{st} - {et}]")

        answer_text = f"According to the video transcript: {top_chunk['text']}"
        return {
            "answer": answer_text,
            "evidence_raw": time_fmt,
            "evidence_timestamps": [{
                "chunk_id": top_chunk["chunk_id"],
                "start": st,
                "end": et,
                "formatted": time_fmt,
            }],
            "raw_response": f"Answer: {answer_text}\nEvidence: {time_fmt}",
        }
