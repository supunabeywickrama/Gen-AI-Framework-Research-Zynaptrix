"""
init_db.py — Python script to initialize Neon PostgreSQL database schema.
This runs database/schema.sql without needing the 'psql' command line tool.

Usage:
    python -m database.init_db
"""

import os
import logging
from database.neon_vector_store import NeonVectorStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

def init_db():
    schema_path = os.path.join(os.path.dirname(__file__), "schema.sql")
    if not os.path.exists(schema_path):
        logger.error(f"Schema file not found at {schema_path}")
        return

    with open(schema_path, "r", encoding="utf-8") as f:
        sql = f.read()

    store = NeonVectorStore()
    conn = store.connect()
    
    logger.info("Running schema.sql on Neon database...")
    try:
        with conn.cursor() as cur:
            cur.execute(sql)
        conn.commit()
        logger.info("Schema initialized successfully! ✅")
    except Exception as e:
        logger.error(f"Error initializing schema: {e}")
        conn.rollback()
    finally:
        store.close()

if __name__ == "__main__":
    init_db()
