from rag_ingestion.parser import DocumentParser
from rag_ingestion.chunker import ContextualChunker
from rag_embeddings.embedder import embedder
from rag_ingestion.captioner import ImageCaptioner
from rag_db.database import SessionLocal
from rag_db.models import ManualChunk

def process_manual(file_path: str, manual_id: str):
    """
    End-to-End ingestion pipeline workflow (v2 with Vision LLM Captioning).
    """
    parser = DocumentParser()
    chunker = ContextualChunker(chunk_size=500, overlap=100)
    captioner = ImageCaptioner()
    
    # 1. Parse manual (PDF -> Images, Text, Tables)
    print(f"Parsing manual {manual_id}...")
    parsed_data = parser.parse_pdf(file_path, manual_id)
    
    # 2. Enrich: Generate Vision Descriptions for Images
    for item in parsed_data:
        if item["type"] == "image":
            print(f"Generating caption for image at page {item['page']}...")
            desc = captioner.generate_caption(item["path"])
            item["content"] = f"[IMAGE REFERENCE: {item['path']}]\n[VISUAL DESCRIPTION]: {desc}"
    
    # 3. Chunk data (Contextual chunking)
    print(f"Chunking data for {manual_id}...")
    chunks = chunker.chunk_data(parsed_data, manual_id)
    
    db = SessionLocal()
    
    # 4. Embed and store to Unified Vector DB (OpenAI text-embeddings-3 for EVERYTHING)
    print(f"Embedding and storing {len(chunks)} chunks...")
    try:
        for chunk in chunks:
            # All types (text, table, image) now have meaningful 'content'
            emb = embedder.embed_text(chunk["content"])
            
            db_chunk = ManualChunk(
                manual_id=chunk["manual_id"],
                type=chunk["type"],
                content=chunk["content"],
                path=chunk.get("path"),  # Only populated for type='image'
                embedding=emb,
                page=chunk["page"]
            )
            db.add(db_chunk)
            
        db.commit()
        print(f"Ingestion for {manual_id} complete!")
        return {"status": "success", "chunks_processed": len(chunks)}
    except Exception as e:
        db.rollback()
        print(f"Ingestion failed: {e}")
        raise e
    finally:
        db.close()
