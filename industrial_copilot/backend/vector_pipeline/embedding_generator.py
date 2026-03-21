"""
embedding_generator.py — Generate text embeddings using OpenAI API.

Uses OpenAI text-embedding-3-small (1536-dim output, projected to 768-dim)
or sentence-transformers as fallback (offline, no API required).

Config:
  OPENAI_API_KEY in .env       → uses OpenAI (recommended, cloud)
  If key is missing or "USE_LOCAL=true" → falls back to sentence-transformers

Usage:
    from vector_pipeline.embedding_generator import EmbeddingGenerator
    gen = EmbeddingGenerator()
    vectors = gen.embed_chunks(chunks)   # List[Dict] → adds 'embedding' key
    vector  = gen.embed_query("motor overload fault")
"""

import os
import time
import logging
from typing import List, Dict, Any, Optional

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
USE_LOCAL      = os.getenv("USE_LOCAL_EMBEDDINGS", "false").lower() == "true"

# Embedding dimensions
OPENAI_DIM = 768    # We request 768-dim from OpenAI (dimensions parameter)
LOCAL_DIM  = 768    # all-mpnet-base-v2 outputs 768-dim natively


# ─────────────────────────────────────────────────────────────────────────────
# EmbeddingGenerator
# ─────────────────────────────────────────────────────────────────────────────
class EmbeddingGenerator:
    """
    Generates text embeddings for document chunks and queries.

    Automatically selects backend:
      - OpenAI text-embedding-3-small (if OPENAI_API_KEY is set)
      - sentence-transformers/all-mpnet-base-v2 (offline fallback)

    Both backends produce 768-dimensional vectors, matching the pgvector schema.
    """

    def __init__(self):
        self._backend: str = self._detect_backend()
        self._model = None
        logger.info(f"[EmbeddingGenerator] Backend: {self._backend}")
        print(f"[EmbeddingGenerator] Using backend: {self._backend}")

    def _detect_backend(self) -> str:
        if USE_LOCAL:
            return "local"
        if OPENAI_API_KEY and not OPENAI_API_KEY.startswith("sk-proj-YOUR"):
            return "openai"
        return "local"

    # ── OpenAI backend ───────────────────────────────────────────────────────
    def _embed_openai(self, texts: List[str]) -> List[List[float]]:
        """Call OpenAI Embeddings API with batch support and rate-limit retry."""
        try:
            import openai
        except ImportError:
            raise ImportError("pip install openai>=1.0 to use OpenAI embeddings.")

        client = openai.OpenAI(api_key=OPENAI_API_KEY)
        batch_size = 100
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i: i + batch_size]
            for attempt in range(3):
                try:
                    response = client.embeddings.create(
                        model="text-embedding-3-small",
                        input=batch,
                        dimensions=OPENAI_DIM,
                    )
                    batch_embeddings = [item.embedding for item in response.data]
                    all_embeddings.extend(batch_embeddings)
                    logger.debug(f"OpenAI batch {i // batch_size + 1}: "
                                 f"{len(batch)} texts embedded.")
                    break
                except Exception as e:
                    if attempt == 2:
                        raise
                    logger.warning(f"OpenAI rate limit — retry {attempt + 1}/3: {e}")
                    time.sleep(2 ** attempt)

        return all_embeddings

    # ── Local (sentence-transformers) backend ────────────────────────────────
    def _load_local_model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                print("[EmbeddingGenerator] Loading sentence-transformers model "
                      "(first run downloads ~420 MB)…")
                self._model = SentenceTransformer("all-mpnet-base-v2")
                print("[EmbeddingGenerator] Model loaded ✅")
            except ImportError:
                raise ImportError(
                    "pip install sentence-transformers to use local embeddings."
                )

    def _embed_local(self, texts: List[str]) -> List[List[float]]:
        self._load_local_model()
        embeddings = self._model.encode(texts, show_progress_bar=True,
                                        batch_size=32)
        return embeddings.tolist()

    # ── Public API ───────────────────────────────────────────────────────────
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a list of text strings.

        Args:
            texts: List of strings to embed.

        Returns:
            List of 768-dimensional float vectors.
        """
        if self._backend == "openai":
            return self._embed_openai(texts)
        return self._embed_local(texts)

    def embed_query(self, query: str) -> List[float]:
        """
        Embed a single query string.

        Args:
            query: The search query text.

        Returns:
            768-dimensional float vector.
        """
        result = self.embed_texts([query])
        return result[0]

    def embed_chunks(
        self, chunks: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Add 'embedding' key to each chunk dict in-place.

        Args:
            chunks: List of chunk dicts from DocumentParser
                    (must have 'content' key).

        Returns:
            Same list of chunks, each now also containing 'embedding'.
        """
        texts = [c["content"] for c in chunks]
        print(f"[EmbeddingGenerator] Embedding {len(texts)} chunks…")
        embeddings = self.embed_texts(texts)
        for chunk, emb in zip(chunks, embeddings):
            chunk["embedding"] = emb
        print(f"[EmbeddingGenerator] Done — {len(chunks)} embeddings generated ✅")
        return chunks
