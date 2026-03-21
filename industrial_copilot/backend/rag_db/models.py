from sqlalchemy import Column, Integer, String, Text
from pgvector.sqlalchemy import Vector
from rag_db.database import Base

class ManualChunk(Base):
    __tablename__ = "manual_chunks"
    
    id = Column(Integer, primary_key=True, index=True)
    manual_id = Column(String, index=True, nullable=False)
    type = Column(String, nullable=False) # 'text', 'image', 'table'
    content = Column(Text, nullable=True) # Text content or structured table string
    embedding = Column(Vector, nullable=False) # Dimension omitted to allow 1536 (OpenAI text) or 512 (CLIP image)
    page = Column(Integer, nullable=True)
    path = Column(String, nullable=True) # Path to the extracted image file
