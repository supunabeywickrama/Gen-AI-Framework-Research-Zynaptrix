from sqlalchemy import create_engine, text
import sys
import os

# Add the parent directory to sys.path to import unified_rag
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from unified_rag.config import settings

def migrate():
    print(f"Connecting to: {settings.database_url.split('@')[-1]}") # Log host only for safety
    engine = create_engine(settings.database_url)
    with engine.connect() as conn:
        try:
            conn.execute(text("ALTER TABLE chat_history ADD COLUMN IF NOT EXISTS message_metadata TEXT"))
            conn.commit()
            print("Successfully checked/added message_metadata column.")
        except Exception as e:
            print(f"Migration error: {e}")

if __name__ == "__main__":
    migrate()
