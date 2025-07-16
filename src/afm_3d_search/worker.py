# src/afm_3d_search/worker.py
import time
import subprocess
import json
from pathlib import Path
import shutil

JOBS_PENDING_DIR = Path("jobs/pending")
JOBS_PROCESSING_DIR = Path("jobs/processing")
JOBS_COMPLETED_DIR = Path("jobs/completed")
JOBS_FAILED_DIR = Path("jobs/failed")

def run_pipeline_for_job(job_path: Path):
    """Executes the main ML pipeline as a subprocess."""
    scene_id = job_path.stem
    print(f"🚀 Processing job for scene: {scene_id}")
    
    command = [
        "python",
        "src/afm_3d_search/run_pipeline.py",
        f"scene_id={scene_id}",
        "paths.raw_dir_name=staging"
    ]
    
    try:
        # We use check=True to raise an exception if the script fails
        subprocess.run(command, check=True, text=True, capture_output=True)
        print(f"✅ Successfully completed job for scene: {scene_id}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Job failed for scene: {scene_id}")
        print(f"--- STDOUT ---\n{e.stdout}")
        print(f"--- STDERR ---\n{e.stderr}")
        return False

def main():
    """Main worker loop to poll for and process jobs."""
    print("🤖 Worker started. Polling for jobs...")
    
    # Ensure directories exist
    for d in [JOBS_PROCESSING_DIR, JOBS_COMPLETED_DIR, JOBS_FAILED_DIR]:
        d.mkdir(parents=True, exist_ok=True)
        
    while True:
        try:
            # Find the first available job
            pending_jobs = sorted(list(JOBS_PENDING_DIR.glob("*.json")))
            if not pending_jobs:
                time.sleep(5)  # Wait if no jobs are available
                continue
                
            job_path = pending_jobs[0]
            processing_path = JOBS_PROCESSING_DIR / job_path.name
            shutil.move(job_path, processing_path)
            
            # Run the job
            success = run_pipeline_for_job(processing_path)
            
            # Move job ticket to its final destination
            if success:
                shutil.move(processing_path, JOBS_COMPLETED_DIR / job_path.name)
            else:
                shutil.move(processing_path, JOBS_FAILED_DIR / job_path.name)

        except Exception as e:
            print(f"An unexpected error occurred in the worker loop: {e}")
            time.sleep(15) # Wait a bit longer after an error

if __name__ == "__main__":
    main()