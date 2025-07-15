# src/afm_3d_search/api/main.py
import uuid
import shutil
from pathlib import Path
import json
from fastapi import FastAPI, UploadFile, File, HTTPException
from typing import List

app = FastAPI(title="3D Scene Processing API")

STAGING_DIR = Path("data/staging")
JOBS_PENDING_DIR = Path("jobs/pending")

@app.on_event("startup")
def on_startup():
    """Ensure necessary directories exist when the app starts."""
    STAGING_DIR.mkdir(exist_ok=True)
    JOBS_PENDING_DIR.mkdir(exist_ok=True)

@app.post("/v1/scenes", status_code=202)
async def create_processing_job(images: List[UploadFile] = File(...)):
    """
    Accepts multiple image files, saves them, and creates a job for background processing.
    """
    if not images:
        raise HTTPException(status_code=400, detail="No images were uploaded.")
    
    scene_id = str(uuid.uuid4())
    scene_staging_dir = STAGING_DIR / scene_id
    scene_staging_dir.mkdir()

    # Save all uploaded images
    for image in images:
        try:
            with open(scene_staging_dir / image.filename, "wb") as buffer:
                shutil.copyfileobj(image.file, buffer)
        finally:
            image.file.close()
    
    # Create the job ticket
    job_ticket = {
        "scene_id": scene_id,
        "image_count": len(images),
        "status": "pending"
    }
    with open(JOBS_PENDING_DIR / f"{scene_id}.json", "w") as f:
        json.dump(job_ticket, f)
        
    print(f"✅ Job created for scene: {scene_id}")

    return {"message": "Job created successfully. Processing in the background.", "scene_id": scene_id}