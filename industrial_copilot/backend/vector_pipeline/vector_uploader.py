"""
vector_uploader.py — Upload embedded document chunks to Neon pgvector.

This is the indexing pipeline entry point. Run it once to seed the knowledge
base, and re-run after adding new maintenance documents.

Usage (CLI):
    python -m vector_pipeline.vector_uploader

Usage (Python):
    from vector_pipeline.vector_uploader import VectorUploader
    uploader = VectorUploader()
    uploader.run()
"""

import logging
import sys
from typing import List, Dict, Any

from vector_pipeline.document_parser import DocumentParser
from vector_pipeline.embedding_generator import EmbeddingGenerator
from database.neon_vector_store import NeonVectorStore

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# VectorUploader
# ─────────────────────────────────────────────────────────────────────────────
class VectorUploader:
    """
    Orchestrates the full indexing pipeline:
        1. Parse documents → chunks
        2. Generate embeddings
        3. Upload to Neon pgvector
    """

    def __init__(
        self,
        clear_existing: bool = False,
        chunk_size: int = 800,
        chunk_overlap: int = 150,
    ):
        """
        Args:
            clear_existing:  If True, deletes all existing docs before re-indexing.
            chunk_size:      Characters per chunk.
            chunk_overlap:   Overlap between consecutive chunks.
        """
        self.parser    = DocumentParser(chunk_size=chunk_size,
                                        chunk_overlap=chunk_overlap)
        self.embedder  = EmbeddingGenerator()
        self.store     = NeonVectorStore()
        self.clear_existing = clear_existing

    def run(self, extra_files: List[Dict[str, Any]] = None) -> int:
        """
        Run the full indexing pipeline.

        Args:
            extra_files: Optional list of extra file dicts to index alongside
                         built-in knowledge:
                         [{"path": "...", "fault_type": "...", "sensor": "..."}]

        Returns:
            Number of chunks successfully uploaded.
        """
        self.store.connect()

        # Optionally clear existing data before re-indexing
        if self.clear_existing:
            existing = self.store.count_documents()
            if existing > 0:
                print(f"[VectorUploader] Clearing {existing} existing documents…")
                self.store.clear_documents()

        # Step 1: Parse built-in knowledge
        print("[VectorUploader] ── Step 1/3: Parsing documents…")
        chunks = self.parser.parse_builtin_knowledge()

        # Step 1b: Parse any additional files provided
        if extra_files:
            for file_info in extra_files:
                path = file_info.get("path", "")
                fault_type = file_info.get("fault_type")
                sensor = file_info.get("sensor")
                try:
                    file_chunks = self.parser.parse_file(path, fault_type=fault_type,
                                                          sensor=sensor)
                    chunks.extend(file_chunks)
                    print(f"[VectorUploader] Added {len(file_chunks)} chunks "
                          f"from '{path}'")
                except FileNotFoundError:
                    logger.warning(f"File not found, skipping: {path}")

        total_chunks = len(chunks)
        print(f"[VectorUploader] Total chunks to index: {total_chunks}")

        # Step 2: Generate embeddings
        print("[VectorUploader] ── Step 2/3: Generating embeddings…")
        chunks = self.embedder.embed_chunks(chunks)

        # Step 3: Upload to Neon
        print("[VectorUploader] ── Step 3/3: Uploading to Neon pgvector…")
        inserted = self.store.insert_document_chunks_batch(chunks)

        final_count = self.store.count_documents()
        print(f"\n[VectorUploader] ✅ Done!")
        print(f"  Inserted this run : {inserted}")
        print(f"  Total in store    : {final_count}")
        print(f"  Vector dimension  : 768")

        self.store.close()
        return inserted

    def status(self):
        """Print current status of the vector store."""
        with NeonVectorStore() as store:
            count = store.count_documents()
            print(f"\n[VectorUploader] Neon Vector Store Status:")
            print(f"  Total document chunks : {count}")
            print(f"  {'Ready for queries ✅' if count > 0 else 'Empty — run uploader first ⚠️'}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Upload machine maintenance documents to Neon pgvector knowledge base."
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Clear existing documents before re-indexing (full re-index).",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Show current vector store status and exit.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    uploader = VectorUploader(clear_existing=args.clear)

    if args.status:
        uploader.status()
        sys.exit(0)

    uploader.run()
