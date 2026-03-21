from sqlalchemy.orm import Session
from rag_db.models import ManualChunk
from rag_embeddings.embedder import embedder

class RetrievalEngine:
    def __init__(self, top_k_text=3, top_k_image=1):
        self.top_k_text = top_k_text
        self.top_k_image = top_k_image
        
    def retrieve(self, db: Session, query: str, manual_id: str):
        """
        Unified Vector Search:
        Images are captioned via Vision LLMs, meaning their 'description'
        exists in the same 1536-dim OpenAI embedding space as regular text!
        """
        # 1. Embed query once using OpenAI text embedder
        query_emb = embedder.embed_text(query)
        
        # 2. Filter by manual_id & type -> Search Text/Tables
        try:
            text_results = db.query(ManualChunk).filter(
                ManualChunk.manual_id == manual_id,
                ManualChunk.type.in_(["text", "table"])
            ).order_by(
                ManualChunk.embedding.cosine_distance(query_emb)
            ).limit(self.top_k_text).all()
        except Exception as e:
            print("Error retrieving text chunks. Ensure pgvector is active:", e)
            text_results = []
            
        # 3. Filter by manual_id & type -> Search Image Captions
        try:
            image_results = db.query(ManualChunk).filter(
                ManualChunk.manual_id == manual_id,
                ManualChunk.type == "image"
            ).order_by(
                ManualChunk.embedding.cosine_distance(query_emb)
            ).limit(self.top_k_image).all()
        except Exception as e:
            print("Error retrieving image chunks. Ensure pgvector is active:", e)
            image_results = []
            
        return {
            "text_chunks": text_results,
            "images": image_results
        }
