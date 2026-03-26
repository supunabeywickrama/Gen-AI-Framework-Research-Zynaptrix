from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List
from unified_rag.db.database import SessionLocal
from unified_rag.db.models import Machine
from pydantic import BaseModel

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

@router.get("/api/machines", response_model=List[MachineResponse])
async def list_machines(db: Session = Depends(get_db)):
    return db.query(Machine).all()

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
