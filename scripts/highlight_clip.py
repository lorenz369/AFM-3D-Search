# src/afm_3d_search/highlight_clip.py

import hydra
from omegaconf import DictConfig # Can keep DictConfig or use MainConfig
import numpy as np
import trimesh
from sklearn.metrics.pairwise import cosine_similarity
import torch
import clip
from pathlib import Path
from conf.schema import MainConfig

@hydra.main(config_path="conf", config_name="config", version_base=None)
def highlight_by_text(cfg: MainConfig) -> None:
    """Highlights points in a completed scene that match a text query."""
    print(f"🔎 Starting CLIP highlight for scene '{cfg.scene_id}'")
    print(f"Query: '{cfg.highlight.text_query}'")

    # --- Load Data from a completed run ---
    # Construct the path to the pipeline's output directory
    input_dir = Path(cfg.paths.data_root) / cfg.paths.completed_dir_name / cfg.scene_id
    
    ply_file = input_dir / cfg.paths.ply_filename
    features_file = input_dir / cfg.paths.clip_features_filename
    
    if not ply_file.exists() or not features_file.exists():
        print(f"❌ Error: Input files not found in {input_dir}")
        print(f"Please run the main pipeline for scene_id='{cfg.scene_id}' first.")
        return

    print("Loading point cloud and CLIP features...")
    pc = trimesh.load(str(ply_file))
    features = np.load(str(features_file))
    original_points = np.array(pc.vertices)
    original_colors = np.array(pc.colors)
    print(f"Data loaded. Total points: {len(original_points)}")

    # --- Load Model and Process Query ---
    print(f"Loading CLIP model ({cfg.models.clip.version})...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _ = clip.load(cfg.models.clip.version, device=device)

    text = clip.tokenize([cfg.highlight.text_query]).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text)
        text_features /= text_features.norm(dim=-1, keepdim=True)
    query_feature = text_features.cpu().numpy()

    # --- Calculate Similarity and Find Top K ---
    similarities = cosine_similarity(query_feature, features)[0]
    similar_indices = np.argsort(similarities)[-cfg.highlight.top_k:]
    
    # --- Create and Save Visualization ---
    output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    query_slug = cfg.highlight.text_query.lower().replace(" ", "_")
    output_file = output_dir / f"clip_highlight_{query_slug}.ply"
    
    validation_colors = np.mean(original_colors[:, :3], axis=1, keepdims=True).astype(np.uint8)
    validation_colors = np.tile(validation_colors, (1, 3))
    validation_colors[similar_indices] = [0, 255, 0]

    print(f"💾 Saving validation point cloud to {output_file}")
    validation_pc = trimesh.PointCloud(vertices=original_points, colors=validation_colors)
    validation_pc.export(str(output_file))

    print("\n✅ Done! Check the new timestamped folder inside 'outputs' for the result.")

if __name__ == "__main__":
    highlight_by_text()