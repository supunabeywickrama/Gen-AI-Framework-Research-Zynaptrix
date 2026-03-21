"""
neon_config.py — Neon PostgreSQL + pgvector connection configuration.
"""

import os
from dotenv import load_dotenv

load_dotenv()

NEON_DB_URL      = os.getenv("NEON_DB_URL", "postgresql://user:password@host/dbname")
EMBEDDING_DIM    = 768          # Embedding vector size (e.g. sentence-transformers)
VECTOR_TABLE     = "machine_documents"
