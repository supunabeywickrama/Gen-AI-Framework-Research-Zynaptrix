import os
import psycopg2
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY is not set. Please check your .env file.")

client = OpenAI(api_key=api_key)

def search_manual(query, limit=5):
    """Embeds the user query and retrieves the most relevant chunks from PostgreSQL pgvector."""
    print(f"Embedding query: '{query}'...")
    
    # Generate query embedding
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=query
    )
    query_embedding = response.data[0].embedding
    
    try:
        conn = psycopg2.connect(
            dbname="manual_db",
            user="postgres",
            password="your_password",
            host="localhost"
        )
    except Exception as e:
        print(f"Failed to connect to database. Make sure PostgreSQL is running. Error: {e}")
        return []

    cursor = conn.cursor()
    
    print("Searching the database...")
    # Using pgvector cosine distance operator '<=>' or L2 distance '<->'
    # We'll use cosine distance as it's standard for OpenAI embeddings
    cursor.execute("""
        SELECT chunk_id, text, source, 1 - (embedding <=> %s::vector) as similarity
        FROM manual_chunks
        ORDER BY embedding <=> %s::vector
        LIMIT %s;
    """, (query_embedding, query_embedding, limit))
    
    results = cursor.fetchall()
    
    cursor.close()
    conn.close()
    
    # Format the results
    chunks = []
    for row in results:
        chunks.append({
            "chunk_id": row[0],
            "text": row[1],
            "source": row[2],
            "similarity": row[3]
        })
        
    return chunks

if __name__ == "__main__":
    test_query = "ESP32" # Example test query based on chunks.json contents found earlier
    results = search_manual(test_query, limit=3)
    
    print(f"\nTop results for '{test_query}':")
    for r in results:
        print(f"\n--- Chunk {r['chunk_id']} (Sim: {r['similarity']:.3f}) ---")
        print(r['text'][:300] + "...")
