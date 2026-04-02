from sqlalchemy import Column, Integer, String, Text, Boolean, ForeignKey
from pgvector.sqlalchemy import Vector
from unified_rag.db.database import Base

class ManualChunk(Base):
    __tablename__ = "manual_chunks"
    __table_args__ = {'extend_existing': True}
    
    id = Column(Integer, primary_key=True, index=True)
    manual_id = Column(String, index=True, nullable=False)
    type = Column(String, nullable=False) # 'text', 'image', 'table'
    content = Column(Text, nullable=True) # Text content or structured table string
    embedding = Column(Vector, nullable=False) # Dimension 1536 (OpenAI text) or 512 (CLIP image)
    page = Column(Integer, nullable=True)
    path = Column(String, nullable=True) # Path to the extracted image file

class Machine(Base):
    __tablename__ = "machines"
    __table_args__ = {'extend_existing': True}
    
    id = Column(Integer, primary_key=True, index=True)
    machine_id = Column(String, unique=True, index=True, nullable=False)
    name = Column(String, nullable=False)
    location = Column(String, nullable=True)
    manual_id = Column(String, nullable=False) # Maps to ManualChunk.manual_id

class AnomalyRecord(Base):
    __tablename__ = "anomaly_records"
    __table_args__ = {'extend_existing': True}
    
    id = Column(Integer, primary_key=True, index=True)
    machine_id = Column(String, index=True, nullable=False)
    timestamp = Column(String, nullable=False)
    type = Column(String, nullable=False) # e.g. 'PUMP_FAULT'
    score = Column(Integer, nullable=False) # Normalized 0-100 or MSE
    sensor_data = Column(Text, nullable=True) # JSON string of readings
    resolved = Column(Boolean, default=False) # Manual HITL sign-off

class ChatMessage(Base):
    __tablename__ = "chat_history"
    __table_args__ = {'extend_existing': True}
    
    id = Column(Integer, primary_key=True, index=True)
    anomaly_id = Column(Integer, ForeignKey("anomaly_records.id"), nullable=True) # NULL for general chat
    role = Column(String, nullable=False) # 'agent' | 'user'
    content = Column(Text, nullable=False)
    timestamp = Column(String, nullable=False)
    images = Column(Text, nullable=True) # JSON list of URLs for agent responses
    # Use 'message_metadata' in Python to avoid conflict with SQLAlchemy's reserved 'metadata' attribute
    message_metadata = Column(Text, nullable=True) # JSON: procedure state, task completion, etc.

class InteractionMemory(Base):
    """
    Vectorized 'Historical Knowledge' derived from resolved incidents.
    This allows the RAG engine to prioritize previous successful fixes.
    """
    __tablename__ = "interaction_memory"
    __table_args__ = {'extend_existing': True}
    
    id = Column(Integer, primary_key=True, index=True)
    machine_id = Column(String, index=True, nullable=False)
    manual_id = Column(String, nullable=False) # Origin manual
    summary = Column(Text, nullable=False) # Actionable summary (steps performed)
    operator_fix = Column(Text, nullable=True) # Final operator input
    embedding = Column(Vector, nullable=False) # 1536 OpenAI
    timestamp = Column(String, nullable=False)
