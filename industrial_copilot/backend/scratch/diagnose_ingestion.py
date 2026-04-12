import os
import sys
import json
import logging
from sqlalchemy import text
from openai import OpenAI

# Add current directory to path
sys.path.append(os.getcwd())

from unified_rag.db.database import engine
from unified_rag.config import settings

logging.basicConfig(level=logging.INFO)

def check_db():
    print("🔍 Checking Database and Vector Extension...")
    try:
        with engine.connect() as conn:
            # Check if vector extension exists
            res = conn.execute(text("SELECT extname FROM pg_extension WHERE extname = 'vector'"))
            ext = res.fetchone()
            if ext:
                print("✅ [DB] Vector extension is ACTIVE.")
            else:
                print("❌ [DB] Vector extension is MISSING.")
                
            # Check if manual_chunks table exists
            res = conn.execute(text("SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = 'manual_chunks')"))
            exists = res.fetchone()[0]
            if exists:
                print("✅ [DB] 'manual_chunks' table exists.")
            else:
                print("❌ [DB] 'manual_chunks' table is MISSING.")
    except Exception as e:
        print(f"❌ [DB] Error connecting to database: {e}")

def check_openai():
    print("\n🔍 Checking OpenAI API Connection...")
    client = OpenAI(api_key=settings.openai_api_key)
    try:
        # Simple test with gpt-4o-mini
        res = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=5
        )
        print(f"✅ [OpenAI] API is reachable. Response: {res.choices[0].message.content}")
    except Exception as e:
        print(f"❌ [OpenAI] API Error: {e}")

if __name__ == "__main__":
    check_db()
    check_openai()
