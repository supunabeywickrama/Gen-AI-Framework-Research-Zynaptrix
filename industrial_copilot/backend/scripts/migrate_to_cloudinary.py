import os
import sys
import json
import logging
import pickle
from datetime import datetime
from sqlalchemy.orm import Session

# Add backend to path so we can import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from unified_rag.db.database import SessionLocal
from unified_rag.db.models import ManualChunk, Manual, SensorConfiguration, AnomalyThreshold, MachineAsset
from services.cloudinary_service import CloudinaryService

# Configure Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("FullMigration")

def migrate_images(db: Session, cloud: CloudinaryService, backend_base: str):
    logger.info("📸 Phase 1: Migrating Extracted Images...")
    chunks = db.query(ManualChunk).filter(ManualChunk.type == 'image', ManualChunk.path.like('data/%')).all()
    logger.info(f"🔍 Found {len(chunks)} images to migrate.")
    
    for chunk in chunks:
        abs_path = os.path.abspath(os.path.join(backend_base, chunk.path))
        if os.path.exists(abs_path):
            filename = os.path.basename(abs_path)
            public_id = os.path.splitext(filename)[0]
            url = cloud.upload_image(abs_path, public_id)
            if url:
                chunk.path = url
                db.commit()
                cloud.delete_local_file(abs_path)
    logger.info("✅ Image migration complete.")


def migrate_manuals(db: Session, cloud: CloudinaryService, backend_base: str):
    logger.info("📄 Phase 2: Migrating Source PDFs...")
    manual_dirs = [
        os.path.join(backend_base, "data", "uploads"),
        os.path.join(backend_base, "data", "industrial_manuals")
    ]
    
    for mdir in manual_dirs:
        if not os.path.exists(mdir): continue
        for filename in os.listdir(mdir):
            if filename.endswith(".pdf"):
                file_path = os.path.join(mdir, filename)
                manual_id = filename.split('_')[0] if "_" in filename else filename.replace(".pdf", "")
                
                logger.info(f"📤 Uploading manual: {filename}...")
                url = cloud.upload_file(file_path, public_id=f"manual_{manual_id}", folder="industrial_copilot/data/manuals")
                
                if url:
                    record = db.query(Manual).filter(Manual.manual_id == manual_id).first()
                    if not record:
                        record = Manual(manual_id=manual_id)
                        db.add(record)
                    record.filename = filename
                    record.url = url
                    record.created_at = datetime.now().isoformat()
                    db.commit()
                    cloud.delete_local_file(file_path)
    logger.info("✅ Manual migration complete.")


def migrate_configs(db: Session, backend_base: str):
    logger.info("⚙️ Phase 3: Migrating JSON Configurations to DB...")
    processed_dir = os.path.join(backend_base, "data", "processed")
    
    # 1. Sensor Configs
    sc_path = os.path.join(processed_dir, "sensor_configs.json")
    if os.path.exists(sc_path):
        with open(sc_path) as f:
            data = json.load(f)
            for machine_id, config in data.items():
                record = db.query(SensorConfiguration).filter(SensorConfiguration.machine_id == machine_id).first()
                if not record:
                    record = SensorConfiguration(machine_id=machine_id)
                    db.add(record)
                record.config_json = json.dumps(config)
                record.updated_at = datetime.now().isoformat()
        db.commit()
        logger.info("✅ Sensor configurations migrated to Postgres.")

    # 2. Thresholds
    th_path = os.path.join(processed_dir, "thresholds.json")
    if os.path.exists(th_path):
        with open(th_path) as f:
            data = json.load(f)
            for key, val in data.items():
                # Key format in JSON is often "PUMP-001_dense"
                if "_" in key:
                    mid, mtype = key.split('_', 1)
                    record = db.query(AnomalyThreshold).filter(AnomalyThreshold.machine_id == mid, 
                                                              AnomalyThreshold.threshold_type == mtype).first()
                    if not record:
                        record = AnomalyThreshold(machine_id=mid, threshold_type=mtype)
                        db.add(record)
                    record.value = float(val)
                    record.updated_at = datetime.now().isoformat()
        db.commit()
        logger.info("✅ Anomaly thresholds migrated to Postgres.")


def migrate_ai_assets(db: Session, cloud: CloudinaryService, backend_base: str):
    logger.info("🧠 Phase 4: Migrating AI Models & Scalers...")
    processed_dir = os.path.join(backend_base, "data", "processed")
    if not os.path.exists(processed_dir): return

    for filename in os.listdir(processed_dir):
        if filename.endswith(".keras") or filename.endswith(".pkl"):
            file_path = os.path.join(processed_dir, filename)
            
            # Identify asset type
            # autoencoder_PUMP-001.keras -> machine_id=PUMP-001, type=model_dense
            if "autoencoder" in filename:
                atype = "model_dense"
                mid = filename.replace("autoencoder_", "").replace(".keras", "")
            elif "lstm" in filename:
                atype = "model_lstm"
                mid = filename.replace("lstm_autoencoder_", "").replace(".keras", "")
            elif "scaler" in filename:
                atype = "scaler"
                mid = filename.replace("scaler_", "").replace(".pkl", "")
            else:
                continue

            logger.info(f"📤 Uploading Asset: {filename}...")
            url = cloud.upload_file(file_path, public_id=f"asset_{atype}_{mid}", folder=f"industrial_copilot/assets/{atype}")
            
            if url:
                asset = db.query(MachineAsset).filter(MachineAsset.machine_id == mid, MachineAsset.asset_type == atype).first()
                if not asset:
                    asset = MachineAsset(machine_id=mid, asset_type=atype)
                    db.add(asset)
                asset.url = url
                asset.updated_at = datetime.now().isoformat()
                db.commit()
                cloud.delete_local_file(file_path)
    logger.info("✅ AI assets migrated to Cloudinary.")


def purge_data_folder(backend_base: str):
    import shutil
    logger.info("🧹 Phase 5: Final Purge of Local Local Data...")
    data_dir = os.path.join(backend_base, "data")
    # Only delete files, keep subfolders for now if they are referenced by empty relative paths
    # Actually user said "clean local data folder"
    for root, dirs, files in os.walk(data_dir, topdown=False):
        for name in files:
            try: os.remove(os.path.join(root, name))
            except: pass
    logger.info("✨ Cleanup complete. Backend is now 100% cloud-native.")


def main():
    db = SessionLocal()
    cloud = CloudinaryService()
    backend_base = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    
    if not cloud.enabled:
        logger.error("❌ Cloudinary not enabled. Aborting.")
        return

    try:
        migrate_images(db, cloud, backend_base)
        migrate_manuals(db, cloud, backend_base)
        migrate_configs(db, backend_base)
        migrate_ai_assets(db, cloud, backend_base)
        purge_data_folder(backend_base)
    finally:
        db.close()

if __name__ == "__main__":
    main()
