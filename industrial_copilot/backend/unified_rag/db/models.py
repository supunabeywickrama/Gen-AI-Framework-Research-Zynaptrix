from sqlalchemy import Column, Integer, String, Text
from pgvector.sqlalchemy import Vector
from unified_rag.db.database import Base

class ManualChunk(Base):
    __tablename__ = "manual_chunks"
    
    id = Column(Integer, primary_key=True, index=True)
    manual_id = Column(String, index=True, nullable=False)
    type = Column(String, nullable=False) # 'text', 'image', 'table'
    content = Column(Text, nullable=True) # Text content or structured table string
    embedding = Column(Vector, nullable=False) # Dimension omitted to allow 1536 (OpenAI text) or 512 (CLIP image)
    page = Column(Integer, nullable=True)
    path = Column(String, nullable=True) # Path to the extracted image file

class Machine(Base):
    __tablename__ = "machines"
    
    id = Column(Integer, primary_key=True, index=True)
    machine_id = Column(String, unique=True, index=True, nullable=False)
    name = Column(String, nullable=False)
    location = Column(String, nullable=True)
    manual_id = Column(String, nullable=False) # Maps to ManualChunk.manual_id

class AnomalyRecord(Base):
    __tablename__ = "anomaly_records"
    
    id = Column(Integer, primary_key=True, index=True)
    machine_id = Column(String, index=True, nullable=False)
    timestamp = Column(String, nullable=False)
    type = Column(String, nullable=False) # e.g. 'PUMP_FAULT'
    score = Column(Integer, nullable=False) # Normalized 0-100 or MSE
    sensor_data = Column(Text, nullable=True) # JSON string of readings at time of incident
