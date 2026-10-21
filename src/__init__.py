"""
Video RAG — Phase 1: Transcript-Based Video Question Answering
Modular research baseline for video transcript processing, indexing, retrieval, and QA.
"""

from src.video_processor import VideoProcessor
from src.transcription import Transcriber, TranscriptSegment
from src.chunking import TranscriptChunker, TranscriptChunk
from src.embeddings import BGEEmbedder
from src.vector_store import FAISSVectorStore
from src.retriever import TranscriptRetriever
from src.llm import GroundedLLMGenerator
from src.evaluation import RAGEvaluator
from src.rag_pipeline import VideoRAGPipeline

__all__ = [
    "VideoProcessor",
    "Transcriber",
    "TranscriptSegment",
    "TranscriptChunker",
    "TranscriptChunk",
    "BGEEmbedder",
    "FAISSVectorStore",
    "TranscriptRetriever",
    "GroundedLLMGenerator",
    "RAGEvaluator",
    "VideoRAGPipeline",
]
