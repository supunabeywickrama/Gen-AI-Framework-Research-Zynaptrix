"""
add_ai_validation_columns.py — Database migration for AI Validation Layer.

This migration adds the following columns to the anomaly_records table:
- ai_validation_status: VARCHAR - TRUE_FAULT, SENSOR_GLITCH, or NORMAL_WEAR
- fault_category: VARCHAR - mechanical, thermal, electrical, process, sensor
- ai_confidence_score: FLOAT - 0.0 to 1.0
- ai_engineering_notes: TEXT - AI reasoning explanation

Run this migration after updating the models.py file.

Usage:
    cd backend
    python scripts/add_ai_validation_columns.py
"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sqlalchemy import create_engine, text
from unified_rag.config import settings
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_migration():
    """Add AI validation columns to anomaly_records table."""
    
    print(f"Connecting to: {settings.database_url.split('@')[-1]}")  # Log host only for safety
    engine = create_engine(settings.database_url)
    
    columns_to_add = [
        ("ai_validation_status", "VARCHAR"),
        ("fault_category", "VARCHAR"),
        ("ai_confidence_score", "FLOAT"),
        ("ai_engineering_notes", "TEXT"),
    ]
    
    with engine.connect() as conn:
        for column_name, column_type in columns_to_add:
            try:
                # Use IF NOT EXISTS for PostgreSQL compatibility
                alter_query = text(f"""
                    ALTER TABLE anomaly_records 
                    ADD COLUMN IF NOT EXISTS {column_name} {column_type}
                """)
                conn.execute(alter_query)
                logger.info(f"✅ Checked/added column: {column_name} ({column_type})")
                    
            except Exception as e:
                logger.error(f"❌ Error with column {column_name}: {e}")
        
        conn.commit()
        logger.info("✅ AI Validation columns migration completed successfully!")


if __name__ == "__main__":
    logger.info("🚀 Starting AI Validation columns migration...")
    run_migration()


