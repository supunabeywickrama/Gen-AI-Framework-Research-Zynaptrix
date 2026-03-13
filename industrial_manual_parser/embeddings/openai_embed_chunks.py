import json
import os
import psycopg2
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Ensure API key is present
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY environment variable is not set. Please check your .env file.")

client = OpenAI(api_key=api_key)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHUNKS_FILE = os.path.join(PROJECT_ROOT, "data", "chunks.json")

def setup_database(cursor):
    """Creates the necessary table with pgvector if it doesn't exist."""
    print("Setting up database table...")
    # Enable pgvector extension
    cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS manual_chunks (
            id SERIAL PRIMARY KEY,
            chunk_id INT,
            text TEXT,
            source TEXT,
            start_char INT,
            end_char INT,
            embedding VECTOR(1536)
        );
    """)
    # Clear existing data to avoid duplicates if re-run
    cursor.execute("TRUNCATE TABLE manual_chunks;")

def embed_and_insert_chunks():
    print(f"Loading chunks from {CHUNKS_FILE}...")
    if not os.path.exists(CHUNKS_FILE):
        raise FileNotFoundError(f"Chunks file not found: {CHUNKS_FILE}")
        
    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    print("Connecting to PostgreSQL database...")
    try:
        conn = psycopg2.connect(
            dbname="manual_db",
            user="postgres",
            password="your_password",
            host="localhost",
            port="5432"
        )
    except Exception as e:
        print(f"Failed to connect to database. Make sure PostgreSQL is running. Error: {e}")
        return

    cursor = conn.cursor()
    
    # Setup DB
    setup_database(cursor)
    conn.commit()

    print(f"Found {len(chunks)} chunks. Generating embeddings...")
    
    for i, chunk in enumerate(chunks):
        if i % 10 == 0:
            print(f"Processing chunk {i + 1}/{len(chunks)}...")
            
        try:
            # Generate embedding using OpenAI
            response = client.embeddings.create(
                model="text-embedding-3-small",
                input=chunk["text"]
            )
            embedding = response.data[0].embedding

            # Insert into database
            cursor.execute(
                """
                INSERT INTO manual_chunks
                (chunk_id, text, source, start_char, end_char, embedding)
                VALUES (%s, %s, %s, %s, %s, %s)
                """,
                (
                    chunk["chunk_id"],
                    chunk["text"],
                    chunk["source"],
                    chunk["start_char"],
                    chunk["end_char"],
                    embedding
                )
            )
        except Exception as e:
            print(f"Error processing chunk {chunk['chunk_id']}: {e}")
            conn.rollback()
            return

    # Commit transactions
    conn.commit()
    cursor.close()
    conn.close()
    
    print(f"Successfully embedded and inserted {len(chunks)} chunks into the database!")

if __name__ == "__main__":
    embed_and_insert_chunks()
