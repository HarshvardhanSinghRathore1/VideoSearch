"""
Vector Store Module
Implements FAISS vector indexing with metadata persistence for transcript chunks.
Uses Inner Product (IP) index on L2-normalized embeddings to compute exact Cosine Similarity.
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("FAISSVectorStore")


class FAISSVectorStore:
    """FAISS-backed vector database for storing and querying chunk embeddings."""

    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self.chunks_metadata: List[Dict[str, Any]] = []
        self._index = None
        self._init_index()

    def _init_index(self):
        """Initializes FAISS IndexFlatIP (Inner Product = Cosine on normalized vectors)."""
        try:
            import faiss
            self._index = faiss.IndexFlatIP(self.dimension)
            logger.info(f"Initialized FAISS IndexFlatIP with dimension {self.dimension}.")
        except ImportError:
            raise ImportError(
                "FAISS is not installed. Please install via `pip install faiss-cpu`."
            )

    @property
    def total_vectors(self) -> int:
        """Returns the number of indexed vectors."""
        return self._index.ntotal if self._index is not None else 0

    def add_chunks(
        self,
        chunks: List[Any],  # List[TranscriptChunk] or List[Dict]
        embeddings: np.ndarray,
    ):
        """
        Adds chunk vectors and metadata to the FAISS index.

        Args:
            chunks: List of TranscriptChunk instances or dicts.
            embeddings: Numpy array of shape (N, dimension), float32.
        """
        if len(chunks) == 0:
            logger.warning("No chunks provided to add_chunks.")
            return

        if embeddings.shape[0] != len(chunks):
            raise ValueError(
                f"Mismatch: {embeddings.shape[0]} embeddings provided for {len(chunks)} chunks."
            )
        if embeddings.shape[1] != self.dimension:
            raise ValueError(
                f"Embedding dimension {embeddings.shape[1]} does not match index dimension {self.dimension}."
            )

        vectors = np.ascontiguousarray(embeddings, dtype=np.float32)
        self._index.add(vectors)

        for chunk in chunks:
            if hasattr(chunk, "to_dict"):
                self.chunks_metadata.append(chunk.to_dict())
            else:
                self.chunks_metadata.append(dict(chunk))

        logger.info(f"Added {len(chunks)} vectors to FAISS index. Total vectors: {self.total_vectors}")

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Searches the index for top_k nearest neighbors.

        Args:
            query_embedding: Numpy array of shape (1, dimension) or (dimension,).
            top_k: Number of nearest chunks to retrieve.

        Returns:
            List of dictionaries containing chunk metadata and 'score' (cosine similarity).
        """
        if self.total_vectors == 0:
            logger.warning("Search called on empty vector index.")
            return []

        q_vec = np.ascontiguousarray(query_embedding, dtype=np.float32)
        if len(q_vec.shape) == 1:
            q_vec = np.expand_dims(q_vec, axis=0)

        k = min(top_k, self.total_vectors)
        distances, indices = self._index.search(q_vec, k)

        results: List[Dict[str, Any]] = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue
            meta = dict(self.chunks_metadata[idx])
            meta["score"] = round(float(dist), 4)
            meta["faiss_index"] = int(idx)
            results.append(meta)

        return results

    def save(self, index_path: str | Path, metadata_path: str | Path):
        """Persists the FAISS binary index and chunk metadata JSON."""
        import faiss

        idx_p = Path(index_path)
        meta_p = Path(metadata_path)
        idx_p.parent.mkdir(parents=True, exist_ok=True)
        meta_p.parent.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self._index, str(idx_p.resolve()))

        with open(meta_p, "w", encoding="utf-8") as f:
            json.dump({
                "dimension": self.dimension,
                "total_vectors": self.total_vectors,
                "chunks": self.chunks_metadata,
            }, f, indent=2, ensure_ascii=False)

        logger.info(f"Saved FAISS index to {idx_p} and metadata to {meta_p}")

    def load(self, index_path: str | Path, metadata_path: str | Path):
        """Loads a saved FAISS index and chunk metadata."""
        import faiss

        idx_p = Path(index_path)
        meta_p = Path(metadata_path)

        if not idx_p.exists() or not meta_p.exists():
            raise FileNotFoundError(f"Index or metadata file not found: {idx_p}, {meta_p}")

        self._index = faiss.read_index(str(idx_p.resolve()))
        with open(meta_p, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.dimension = data.get("dimension", self._index.d)
        self.chunks_metadata = data.get("chunks", [])

        logger.info(
            f"Loaded FAISS index with {self.total_vectors} vectors (dim={self.dimension}) "
            f"and {len(self.chunks_metadata)} chunk metadata records."
        )
