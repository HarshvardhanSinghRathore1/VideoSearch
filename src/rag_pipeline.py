"""
RAG Pipeline Module
Orchestrates the entire Video RAG Phase 1 workflow:
Video -> Audio -> Whisper Transcript -> Timestamped Chunks -> BGE Embeddings -> FAISS Index -> LLM QA.
Also provides Baseline A (Full Transcript Context) vs Baseline B (FAISS RAG Retrieval) comparison.
"""

import time
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

from src.video_processor import VideoProcessor
from src.transcription import Transcriber, TranscriptSegment
from src.chunking import TranscriptChunker, TranscriptChunk
from src.embeddings import BGEEmbedder
from src.vector_store import FAISSVectorStore
from src.retriever import TranscriptRetriever
from src.llm import GroundedLLMGenerator
from src.evaluation import RAGEvaluator

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("VideoRAGPipeline")


class VideoRAGPipeline:
    """Master orchestrator for Video RAG Phase 1 research baseline."""

    def __init__(
        self,
        video_processor: Optional[VideoProcessor] = None,
        transcriber: Optional[Transcriber] = None,
        chunker: Optional[TranscriptChunker] = None,
        embedder: Optional[BGEEmbedder] = None,
        vector_store: Optional[FAISSVectorStore] = None,
        llm_generator: Optional[GroundedLLMGenerator] = None,
    ):
        self.video_processor = video_processor or VideoProcessor()
        self.transcriber = transcriber or Transcriber()
        self.chunker = chunker or TranscriptChunker()
        self.embedder = embedder or BGEEmbedder()
        self.vector_store = vector_store or FAISSVectorStore(dimension=self.embedder.dimension)
        self.retriever = TranscriptRetriever(self.embedder, self.vector_store)
        self.llm_generator = llm_generator or GroundedLLMGenerator()

        self.current_video: Optional[str] = None
        self.segments: List[TranscriptSegment] = []
        self.chunks: List[TranscriptChunk] = []

    def process_video_to_transcript(
        self,
        video_path: str | Path,
        save_transcript: bool = True,
    ) -> List[TranscriptSegment]:
        """
        Step 1 & 2: Extract audio from video and transcribe with timestamps.

        Args:
            video_path: Path to raw video file.
            save_transcript: Whether to save JSON and TXT transcript files.

        Returns:
            List of timestamped TranscriptSegment instances.
        """
        v_path = Path(video_path)
        self.current_video = v_path.name

        # 1. Audio Extraction
        audio_path = self.video_processor.extract_audio(v_path)

        # 2. Timestamped Transcription
        self.segments = self.transcriber.transcribe(audio_path)

        # 3. Save Artifacts
        if save_transcript:
            self.transcriber.save_transcripts(self.segments, self.current_video)

        return self.segments

    def index_transcript(
        self,
        segments: Optional[List[TranscriptSegment]] = None,
        video_name: Optional[str] = None,
    ) -> List[TranscriptChunk]:
        """
        Step 3 & 4: Chunk transcript segments and build FAISS vector index.

        Args:
            segments: List of TranscriptSegment instances (uses self.segments if None).
            video_name: Identifier of video.

        Returns:
            List of generated TranscriptChunk objects.
        """
        segs = segments if segments is not None else self.segments
        v_name = video_name or self.current_video or "video"

        if not segs:
            raise ValueError("No transcript segments available to chunk and index.")

        # 1. Timestamp-Preserving Chunking
        self.chunks = self.chunker.chunk_segments(segs, source_video=v_name)

        # 2. Embedding Generation
        embeddings = self.embedder.embed_chunks(self.chunks)

        # 3. Vector Store Ingestion
        self.vector_store.add_chunks(self.chunks, embeddings)

        return self.chunks

    def query(
        self,
        question: str,
        top_k: int = 5,
    ) -> Dict[str, Any]:
        """
        Primary Baseline B (Transcript RAG):
        Retrieves top-k relevant chunks and generates evidence-grounded answer with timestamp citations.

        Args:
            question: Natural language question.
            top_k: Number of retrieved chunks.

        Returns:
            Dictionary with 'answer', 'evidence_timestamps', 'retrieved_chunks', 'latencies', etc.
        """
        t0 = time.time()
        retrieved_chunks = self.retriever.retrieve(question, top_k=top_k)
        retrieval_latency = time.time() - t0

        context_str = self.retriever.format_context_for_llm(retrieved_chunks)

        t1 = time.time()
        llm_response = self.llm_generator.generate_answer(
            query=question,
            retrieved_chunks=retrieved_chunks,
            context_str=context_str,
        )
        generation_latency = time.time() - t1

        return {
            "question": question,
            "pipeline_type": "Baseline_B_Transcript_RAG",
            "answer": llm_response["answer"],
            "evidence_raw": llm_response.get("evidence_raw", ""),
            "evidence_timestamps": llm_response.get("evidence_timestamps", []),
            "retrieved_chunks": retrieved_chunks,
            "context_str": context_str,
            "model": llm_response.get("model", ""),
            "retrieval_latency_sec": round(retrieval_latency, 3),
            "generation_latency_sec": round(generation_latency, 3),
            "total_latency_sec": round(retrieval_latency + generation_latency, 3),
        }

    def query_baseline_a_full_transcript(
        self,
        question: str,
    ) -> Dict[str, Any]:
        """
        Comparative Baseline A:
        Feeds the ENTIRE video transcript directly to the LLM without vector retrieval.

        Args:
            question: Natural language question.

        Returns:
            Dictionary with answer and execution metrics.
        """
        t0 = time.time()

        # Reconstruct full transcript context
        full_context_blocks = []
        for i, seg in enumerate(self.segments, 1):
            from src.transcription import format_timestamp
            full_context_blocks.append(
                f"[{format_timestamp(seg.start)} - {format_timestamp(seg.end)}] {seg.text}"
            )
        full_context_str = "\n".join(full_context_blocks)

        llm_response = self.llm_generator.generate_answer(
            query=question,
            retrieved_chunks=[],
            context_str=full_context_str,
        )
        total_latency = time.time() - t0

        return {
            "question": question,
            "pipeline_type": "Baseline_A_Full_Transcript",
            "answer": llm_response["answer"],
            "evidence_raw": llm_response.get("evidence_raw", ""),
            "evidence_timestamps": llm_response.get("evidence_timestamps", []),
            "retrieved_chunks": [],
            "context_str": full_context_str,
            "model": llm_response.get("model", ""),
            "retrieval_latency_sec": 0.0,
            "generation_latency_sec": round(total_latency, 3),
            "total_latency_sec": round(total_latency, 3),
        }

    def run_evaluation(
        self,
        eval_dataset_path: str | Path,
        experiment_name: str = "baseline_results",
        top_k: int = 5,
    ) -> Dict[str, Any]:
        """
        Runs automated evaluation benchmark across the provided questions dataset.

        Args:
            eval_dataset_path: Path to questions JSON.
            experiment_name: Base filename for results.
            top_k: Top K chunks to retrieve.

        Returns:
            Aggregated metrics summary dictionary.
        """
        evaluator = RAGEvaluator()
        questions = evaluator.load_evaluation_dataset(eval_dataset_path)

        sample_results = []
        logger.info(f"Starting evaluation benchmark on {len(questions)} questions...")

        for q in questions:
            t0 = time.time()
            retrieved = self.retriever.retrieve(q["question"], top_k=top_k)
            ret_lat = time.time() - t0

            context = self.retriever.format_context_for_llm(retrieved)

            t1 = time.time()
            llm_out = self.llm_generator.generate_answer(q["question"], retrieved, context)
            gen_lat = time.time() - t1

            sample_res = evaluator.evaluate_sample(
                sample=q,
                retrieved_chunks=retrieved,
                llm_output=llm_out,
                retrieval_latency=ret_lat,
                generation_latency=gen_lat,
            )
            sample_results.append(sample_res)

        config = {
            "video_identifier": self.current_video or "sample.mp4",
            "transcription_model": self.transcriber.model_size,
            "embedding_model": self.embedder.model_name,
            "embedding_dimension": self.embedder.dimension,
            "chunk_size": self.chunker.target_chunk_size,
            "chunk_overlap": self.chunker.chunk_overlap,
            "top_k": top_k,
            "llm_model": self.llm_generator.model_name,
            "eval_dataset_path": str(eval_dataset_path),
            "eval_dataset_version": "1.0",
        }

        aggregated = evaluator.aggregate_metrics(sample_results, config)
        evaluator.save_results(aggregated, experiment_name=experiment_name)
        return aggregated

    def save_index(self, index_dir: str = "data/vector_store"):
        """Saves current FAISS index and chunk metadata."""
        idx_dir = Path(index_dir)
        idx_dir.mkdir(parents=True, exist_ok=True)
        self.vector_store.save(
            index_path=idx_dir / "faiss.index",
            metadata_path=idx_dir / "chunks_metadata.json",
        )

    def load_index(self, index_dir: str = "data/vector_store"):
        """Loads saved FAISS index and chunk metadata."""
        idx_dir = Path(index_dir)
        self.vector_store.load(
            index_path=idx_dir / "faiss.index",
            metadata_path=idx_dir / "chunks_metadata.json",
        )
