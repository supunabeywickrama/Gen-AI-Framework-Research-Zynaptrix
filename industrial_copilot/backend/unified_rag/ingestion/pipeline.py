from unified_rag.ingestion.parser import DocumentParser
from unified_rag.ingestion.chunker import ContextualChunker
from unified_rag.embeddings.embedder import embedder
from unified_rag.ingestion.captioner import ImageCaptioner
from unified_rag.db.database import SessionLocal
from unified_rag.db.models import ManualChunk

def process_manual(file_path: str, manual_id: str):
    """
    End-to-End ingestion pipeline workflow (v2 with Vision LLM Captioning).
    """
    print(f"🛠️ [Pipeline] Initializing ingestion components for {manual_id}...")
    parser = DocumentParser()
    chunker = ContextualChunker(chunk_size=500, overlap=100)
    captioner = ImageCaptioner()
    
    # 1. Parse manual (PDF -> Images, Text, Tables)
    print(f"🔍 [Pipeline] Stage 1/4: Parsing PDF {file_path}...")
    parsed_data = parser.parse_pdf(file_path, manual_id)
    print(f"✅ [Pipeline] Parsing complete. Found {len(parsed_data)} primitive items.")
    
    # 2. Enrich: Generate Vision Descriptions for Images
    image_count = sum(1 for item in parsed_data if item["type"] == "image")
    print(f"🖼️ [Pipeline] Stage 2/4: Generating captions for {image_count} images...")
    
    for i, item in enumerate(parsed_data):
        if item["type"] == "image":
            print(f"   ∟ Processing image {i+1}/{len(parsed_data)} at page {item['page']}...")
            desc = captioner.generate_caption(item["path"])
            item["content"] = f"[IMAGE REFERENCE: {item['path']}]\n[VISUAL DESCRIPTION]: {desc}"
    
    # 3. Chunk data (Contextual chunking)
    print(f"✂️ [Pipeline] Stage 3/4: Creating contextual chunks...")
    chunks = chunker.chunk_data(parsed_data, manual_id)
    print(f"✅ [Pipeline] Created {len(chunks)} searchable chunks.")
    
    # 4. Embed and store to Unified Vector DB
    print(f"🧠 [Pipeline] Stage 4/4: Embedding and storing to database...")
    
    batch_size = 50
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        print(f"   ∟ Processing batch {i//batch_size + 1}/{(len(chunks)-1)//batch_size + 1} ({len(batch)} chunks)...")
        
        # A. Embed the batch (This takes time, no DB connection open yet)
        batch_to_save = []
        for j, chunk in enumerate(batch):
            if j % 10 == 0:
                print(f"      ∟ Embedding {j+1}/{len(batch)}...")
            
            try:
                emb = embedder.embed_text(chunk["content"])
                batch_to_save.append((chunk, emb))
            except Exception as e:
                print(f"      ⚠️ Embedding failed for a chunk: {e}. Skipping this chunk.")
                continue
        
        # B. Save the batch (Open DB connection only for the actual save)
        if not batch_to_save:
            continue
            
        db = SessionLocal()
        try:
            for chunk, emb in batch_to_save:
                db_chunk = ManualChunk(
                    manual_id=chunk["manual_id"],
                    type=chunk["type"],
                    content=chunk["content"],
                    path=chunk.get("path"), 
                    embedding=emb,
                    page=chunk["page"]
                )
                db.add(db_chunk)
            
            db.commit()
            print(f"   💾 [Pipeline] Batch {i//batch_size + 1} saved successfully.")
        except Exception as e:
            db.rollback()
            print(f"   🔥 [Pipeline] Error saving batch {i//batch_size + 1}: {e}")
            # We don't raise here so the process can attempt subsequent batches
        finally:
            db.close()

    print(f"✨ [Pipeline] SUCCESS: Ingestion for {manual_id} finalized.")
    return {"status": "success", "chunks_processed": len(chunks)}
