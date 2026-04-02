from unified_rag.db.database import engine
from sqlalchemy import text

with engine.connect() as conn:
    conn.execute(text("ALTER TABLE chat_history ADD COLUMN IF NOT EXISTS metadata TEXT"))
    conn.commit()
print("ALTER TABLE SUCCESSFUL")
