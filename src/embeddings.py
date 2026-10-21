"""
Embedding Module
Generates dense vector representations for transcript chunks and search queries using
the BAAI/bge-small-en-v1.5 model via sentence-transformers with L2 normalization.
"""

import logging
import numpy as np
from typing import List, Union, Optional
from src.chunking import TranscriptChunk

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("BGEEmbedder")

DEFAULT_MODEL_NAME = "BAAI/bge-small-en-v1.5"
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class BGEEmbedder:
    """Wrapper for sentence-transformers embedding generation with BGE model support."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        device: Optional[str] = None,
        normalize_embeddings: bool = True,
        batch_size: int = 32,
    ):
        self.model_name = model_name
        self.normalize_embeddings = normalize_embeddings
        self.batch_size = batch_size
        self._device = device
        self._model = None
        self._dimension = None

    def _load_model(self):
        """Loads sentence-transformers model lazily."""
        if self._model is not None:
            return

        try:
            from sentence_transformers import SentenceTransformer
            import torch

            if self._device is None:
                self._device = "cuda" if torch.cuda.is_available() else "cpu"

            logger.info(f"Loading embedding model '{self.model_name}' on {self._device}...")
            self._model = SentenceTransformer(self.model_name, device=self._device)
            self._dimension = self._model.get_sentence_embedding_dimension()
            logger.info(f"Embedding model loaded. Vector dimension: {self._dimension}")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise

    @property
    def dimension(self) -> int:
        """Returns embedding vector dimension (e.g. 384 for bge-small-en-v1.5)."""
        self._load_model()
        return self._dimension

    def embed_texts(
        self,
        texts: List[str],
        is_query: bool = False,
        show_progress_bar: bool = False,
    ) -> np.ndarray:
        """
        Embeds a list of raw text strings.

        Args:
            texts: List of strings.
            is_query: If True and using BGE model, prepends query instruction prefix.
            show_progress_bar: Whether to display tqdm progress bar.

        Returns:
            np.ndarray of shape (num_texts, dimension), dtype float32.
        """
        self._load_model()
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)

        formatted_texts = texts
        if is_query and "bge" in self.model_name.lower():
            formatted_texts = [f"{BGE_QUERY_PREFIX}{t}" for t in texts]

        embeddings = self._model.encode(
            formatted_texts,
            batch_size=self.batch_size,
            show_progress_bar=show_progress_bar,
            normalize_embeddings=self.normalize_embeddings,
            convert_to_numpy=True,
        )
        return embeddings.astype(np.float32)

    def embed_chunks(
        self,
        chunks: List[TranscriptChunk],
        show_progress_bar: bool = False,
    ) -> np.ndarray:
        """
        Embeds a list of TranscriptChunk instances.

        Args:
            chunks: List of TranscriptChunk objects.
            show_progress_bar: Progress bar flag.

        Returns:
            np.ndarray of shape (len(chunks), dimension).
        """
        texts = [chunk.text for chunk in chunks]
        return self.embed_texts(texts, is_query=False, show_progress_bar=show_progress_bar)

    def embed_query(self, query: str) -> np.ndarray:
        """
        Embeds a single query string.

        Args:
            query: User question / search query.

        Returns:
            1D or 2D np.ndarray of shape (1, dimension).
        """
        emb = self.embed_texts([query], is_query=True)
        return emb
