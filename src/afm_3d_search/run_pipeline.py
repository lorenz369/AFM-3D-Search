import hydra
from pathlib import Path
import torch
from PIL import Image
import os
from tqdm import tqdm

# Import the refactored modules
from pipeline import reconstruction
from pipeline.feature_extraction import FeatureExtractor
from pipeline import processing

from conf.schema import MainConfig

@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: MainConfig) -> None:
    """Orchestrates the high-speed, hybrid ML pipeline for a given scene."""
    
    # --- 1. Setup ---
    print(f"--- Starting High-Speed Pipeline for Scene: {cfg.scene_id} ---")
    
    data_root = Path(cfg.paths.data_root)
    image_dir = data_root / cfg.paths.raw_dir_name / cfg.scene_id / "images"
    output_dir = data_root / cfg.paths.completed_dir_name / cfg.scene_id
    
    print(f"Reading images from: {image_dir}")
    print(f"Saving results to: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" and torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    image_paths = sorted([str(p) for p in image_dir.glob('*')])
    num_images = len(image_paths)

    if not image_paths:
        raise FileNotFoundError(f"No images found in '{image_dir}'")
    
    # --- 2. Pre-load all images into CPU memory ---
    print(f"Pre-loading {num_images} images into memory...")
    pil_images = [Image.open(p).convert("RGB") for p in image_paths]

    # --- 3. Phase 1: Monolithic Reconstruction (All-at-Once) ---
    vggt_output_gpu = reconstruction.run_vggt(pil_images, device, dtype)
    
    # --- 4. Initialize Feature Extractors Once ---
    feature_extractor = FeatureExtractor(cfg, device)

    # --- 5. Phase 2: Batched GPU-Native Processing ---
    print(f"🚀 Starting batched processing for {num_images} views...")
    
    final_aggregated_data = {
        "points": [], "colors": [], "dino_features": [], "clip_features": []
    }
    
    batch_size = cfg.processing.batch_size
    for i in tqdm(range(0, num_images, batch_size), desc="Processing Batches"):
        
        # --- Create batches for this iteration ---
        batch_end = min(i + batch_size, num_images)
        pil_batch = pil_images[i:batch_end]
        
        # Slice the large GPU tensors from the reconstruction phase
        vggt_batch_gpu = {
            "depth_tensor": vggt_output_gpu["depth_tensor"][:, i:batch_end],
            "confidence_tensor": vggt_output_gpu["confidence_tensor"][:, i:batch_end],
            "images_tensor": vggt_output_gpu["images_tensor"][:, i:batch_end],
            "extrinsic_tensor": vggt_output_gpu["extrinsic_tensor"][:, i:batch_end],
            "intrinsic_tensor": vggt_output_gpu["intrinsic_tensor"][:, i:batch_end],
            "height": vggt_output_gpu["height"],
            "width": vggt_output_gpu["width"]
        }
        
        # --- Extract features for the batch (in-memory) ---
        features_batch_gpu = feature_extractor.run_batch(pil_batch)
        
        # --- Process the batch entirely on GPU ---
        processed_batch_gpu = processing.process_batch_gpu(vggt_batch_gpu, features_batch_gpu, cfg.processing)

        # --- Aggregate results (on CPU to avoid holding a growing tensor in VRAM) ---
        for key in final_aggregated_data.keys():
            if processed_batch_gpu[key] is not None:
                final_aggregated_data[key].append(processed_batch_gpu[key].cpu())

    # Concatenate all batch results
    for key in final_aggregated_data.keys():
        if final_aggregated_data[key]:
            final_aggregated_data[key] = torch.cat(final_aggregated_data[key], dim=0).numpy()
        else:
            # Handle case where no points were aggregated
            final_aggregated_data[key] = None

    # --- 6. Save Final Artifacts ---
    if final_aggregated_data["points"] is not None:
        processing.save_artifacts(output_dir, final_aggregated_data)
        print(f"--- ✅ Successfully Processed Scene: {cfg.scene_id} ---")
    else:
        print(f"--- ⚠️ No points were generated for Scene: {cfg.scene_id}. Nothing to save. ---")


if __name__ == "__main__":
    main()