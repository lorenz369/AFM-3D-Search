import hydra
from omegaconf import DictConfig
from pathlib import Path
import torch
import os
from PIL import Image

# Import the refactored modules
from pipeline import reconstruction, feature_extraction, processing

@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """Orchestrates the entire ML pipeline for a given scene."""
    
    # --- 1. Setup based on explicit config ---
    if "scene_id" not in cfg:
        raise ValueError("A 'scene_id' must be provided via command-line override (e.g., scene_id=bude)")
    
    # NEW: Explicitly define the data source. Defaults to 'staging' for the worker.
    data_source = cfg.get("data_source", "staging")
    if data_source not in ["staging", "testing"]:
        raise ValueError(f"data_source must be 'staging' or 'testing', but got {data_source}")

    scene_id = cfg.scene_id
    image_dir = Path(f"data/{data_source}/{scene_id}/images") if data_source == "testing" else Path(f"data/{data_source}/{scene_id}")
    output_dir = Path(f"data/completed/{scene_id}")
    
    print(f"--- Starting Pipeline for Scene: {scene_id} (Source: {data_source}) ---")
    os.makedirs(output_dir, exist_ok=True)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" and torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    image_paths = sorted([str(p) for p in image_dir.glob('*')])

    if not image_paths:
        raise FileNotFoundError(f"No images found in the expected directory: '{image_dir}'")

    # --- 2. Load Images ---
    print(f"Pre-loading {len(image_paths)} images into memory...")
    pil_images = [Image.open(p).convert("RGB") for p in image_paths]

    # --- 3. Pipeline Execution ---
    # ... (The rest of the script is exactly the same) ...
    vggt_output_gpu = reconstruction.run_vggt(pil_images, device, dtype)
    features_gpu = feature_extraction.run(pil_images, vggt_output_gpu, cfg, device)
    final_data_cpu = processing.filter_and_aggregate(vggt_output_gpu, features_gpu, cfg.processing)
    processing.save_artifacts(output_dir, final_data_cpu)
    
    print(f"--- ✅ Successfully Processed Scene: {scene_id} ---")

if __name__ == "__main__":
    main()