"""
Retriever Module
Connects query embedding with FAISS vector similarity search to return top-k
timestamp-grounded transcript chunks with confidence scores.
"""

import logging
from typing import List, Dict, Any, Optional
from src.embeddings import BGEEmbedder
from src.vector_store import FAISSVectorStore
from src.transcription import format_timestamp

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("TranscriptRetriever")


class TranscriptRetriever:
    """Retrieves top-k relevant transcript chunks for user queries."""

    def __init__(
        self,
        embedder: BGEEmbedder,
        vector_store: FAISSVectorStore,
        default_top_k: int = 5,
    ):
        self.embedder = embedder
        self.vector_store = vector_store
        self.default_top_k = default_top_k

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieves top-k most relevant chunks for a given query.

        Args:
            query: Question or search query text.
            top_k: Number of chunks to retrieve (defaults to self.default_top_k).

        Returns:
            List of chunk dictionaries with similarity scores and timestamps.
        """
        k = top_k if top_k is not None else self.default_top_k
        query = query.strip()
        if not query:
            logger.warning("Empty query passed to retrieve().")
            return []

        # 1. Embed query
        query_vec = self.embedder.embed_query(query)

        # 2. Search FAISS index
        raw_results = self.vector_store.search(query_vec, top_k=k)

        # 3. Format result records
        formatted_results: List[Dict[str, Any]] = []
        for item in raw_results:
            st = float(item["start_time"])
            et = float(item["end_time"])
            formatted_results.append({
                "chunk_id": item["chunk_id"],
                "text": item["text"],
                "start_time": st,
                "end_time": et,
                "score": item["score"],
                "source_video": item.get("source_video", ""),
                "formatted_time": f"[{format_timestamp(st)} - {format_timestamp(et)}]",
            })

        logger.info(f"Retrieved {len(formatted_results)} chunks for query: '{query[:50]}...'")
        return formatted_results

    def format_context_for_llm(self, retrieved_chunks: List[Dict[str, Any]]) -> str:
        """
        Formats retrieved chunks into a clean, numbered context block with timestamps for the LLM.

        Args:
            retrieved_chunks: List of retrieved chunk dictionaries.

        Returns:
            Formatted string block.
        """
        if not retrieved_chunks:
            return "No relevant video transcript context was found."

        context_blocks = []
        for i, chunk in enumerate(retrieved_chunks, 1):
            st_fmt = format_timestamp(chunk["start_time"])
            et_fmt = format_timestamp(chunk["end_time"])
            block = (
                f"[Evidence {i}] (Time: {st_fmt} - {et_fmt} | Chunk: {chunk['chunk_id']}):\n"
                f"{chunk['text'].strip()}"
            )
            context_blocks.append(block)

        return "\n\n".join(context_blocks)
