from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List
from unified_rag.db.database import SessionLocal
from unified_rag.db.models import Machine, AnomalyRecord
from pydantic import BaseModel
import json

router = APIRouter(tags=["Machine Registry"])

# Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Pydantic Schemas
class MachineBase(BaseModel):
    machine_id: str
    name: str
    location: str
    manual_id: str

class MachineResponse(MachineBase):
    class Config:
        from_attributes = True

class AnomalyResponse(BaseModel):
    id: int
    machine_id: str
    timestamp: str
    type: str
    score: int
    sensor_data: str # JSON backend

    class Config:
        from_attributes = True

@router.get("/api/machines", response_model=List[MachineResponse])
async def list_machines(db: Session = Depends(get_db)):
    return db.query(Machine).all()

@router.get("/api/machines/{machine_id}/anomalies", response_model=List[AnomalyResponse])
async def get_machine_anomalies(machine_id: str, db: Session = Depends(get_db)):
    """Fetch history of anomalies for a specific machine."""
    return db.query(AnomalyRecord).filter(AnomalyRecord.machine_id == machine_id).order_by(AnomalyRecord.id.desc()).all()

@router.post("/api/machines", response_model=MachineResponse)
async def register_machine(machine: dict, db: Session = Depends(get_db)):
    """Add or update a machine in the registry."""
    existing = db.query(Machine).filter(Machine.machine_id == machine.get('machine_id')).first()
    if existing:
        for key, value in machine.items():
            setattr(existing, key, value)
        db.commit()
        db.refresh(existing)
        return existing
        
    db_machine = Machine(**machine)
    db.add(db_machine)
    db.commit()
    db.refresh(db_machine)
    return db_machine

@router.post("/api/machines/delete/{machine_id}")
async def decommission_machine(machine_id: str, db: Session = Depends(get_db)):
    """Remove a machine from the registry."""
    machine = db.query(Machine).filter(Machine.machine_id == machine_id).first()
    if not machine:
        raise HTTPException(status_code=404, detail="Asset not found")
    
    db.delete(machine)
    db.commit()
    return {"status": "decommissioned", "machine_id": machine_id}
