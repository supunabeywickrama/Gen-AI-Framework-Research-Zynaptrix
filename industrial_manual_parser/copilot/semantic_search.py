import os
import json
import psycopg2
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY is not set. Please check your .env file.")

client = OpenAI(api_key=api_key)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
LAYOUT_FILE = os.path.join(DATA_DIR, "layout_regions.json")
CROPPED_FOLDER = os.path.join(DATA_DIR, "cropped")
MANUALS_FOLDER = os.path.join(DATA_DIR, "manuals")

def get_related_diagrams(chunks=None, limit=3):
    """Retrieves paths to diagrams on the same pages as the retrieved text chunks."""
    if not os.path.exists(LAYOUT_FILE):
        return []
        
    with open(LAYOUT_FILE, "r") as f:
        regions = json.load(f)
        
    relevant_pages_ordered = []
    
    if chunks:
        # Dynamically map chunks to PDF pages using word overlap
        try:
            import fitz
            pdf_name = next((f for f in os.listdir(MANUALS_FOLDER) if f.endswith(".pdf")), None)
            if pdf_name:
                pdf_path = os.path.join(MANUALS_FOLDER, pdf_name)
                doc = fitz.open(pdf_path)
                
                def get_trigrams(text):
                    words = text.split()
                    return set(" ".join(words[j:j+3]) for j in range(len(words)-2))

                # Pre-extract trigrams for all pages
                page_grams_map = {}
                for i, page in enumerate(doc):
                    page_grams_map[i + 1] = get_trigrams(page.get_text().lower())
                
                for chunk in chunks:
                    chunk_grams = get_trigrams(chunk["text"].lower())
                    best_page = 1
                    best_score = 0
                    for page_num, page_grams in page_grams_map.items():
                        score = len(chunk_grams.intersection(page_grams))
                        if score > best_score:
                            best_score = score
                            best_page = page_num
                            
                    if best_score > 0:
                        page_str = f"page_{best_page}.png"
                        if page_str not in relevant_pages_ordered:
                            relevant_pages_ordered.append(page_str)
        except Exception as e:
            print(f"Failed to load PyMuPDF or link chunk to page: {e}")
        
    diagrams = []
    
    if chunks and relevant_pages_ordered:
        for target_page in relevant_pages_ordered:
            for r in regions:
                if r["page"] == target_page and r.get("type") in ["Picture", "Figure", "Table"]:
                    page_name = os.path.splitext(r["page"])[0]
                    region_id = r["region_id"]
                    block_type = r["type"]
                    region_filename = f"{page_name}_region_{region_id}_{block_type}.png"
                    img_path = os.path.join(CROPPED_FOLDER, region_filename)
                    if os.path.exists(img_path) and img_path not in diagrams:
                        diagrams.append(img_path)
            if len(diagrams) >= limit:
                break
    else:
        # Fallback if no chunks provided
        for r in regions:
            if r.get("type") in ["Picture", "Figure", "Table"]:
                page_name = os.path.splitext(r["page"])[0]
                region_id = r["region_id"]
                block_type = r["type"]
                region_filename = f"{page_name}_region_{region_id}_{block_type}.png"
                img_path = os.path.join(CROPPED_FOLDER, region_filename)
                if os.path.exists(img_path):
                    diagrams.append(img_path)
                    
                if len(diagrams) >= limit:
                    break
                
    return diagrams

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
