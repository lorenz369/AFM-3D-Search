# src/afm_3d_search/highlight_clip.py

import hydra
from omegaconf import DictConfig
import numpy as np
import trimesh
from sklearn.metrics.pairwise import cosine_similarity
import torch
import clip
from pathlib import Path

@hydra.main(config_path="conf", config_name="highlight_clip", version_base=None)
def highlight_by_text(cfg: DictConfig) -> None:
    """
    Loads a point cloud and its CLIP features, and highlights points
    that match a text query.
    """
    print(f"🔎 Starting CLIP highlight for query: '{cfg.text_query}'")

    # Use hydra.utils.to_absolute_path to resolve paths relative to original CWD
    run_dir = Path(hydra.utils.to_absolute_path(cfg.paths.run_output_dir))
    base_name = run_dir.name
    
    ply_file = run_dir / f"{base_name}.ply"
    features_file = run_dir / f"{base_name}_clip_features.npy"
    
    # --- Load Data ---
    if not ply_file.exists() or not features_file.exists():
        print(f"❌ Error: Input files not found in {run_dir}")
        print("Please run the main 'generate_pointcloud.py' script first.")
        return

    print("Loading point cloud data and features...")
    pc = trimesh.load(str(ply_file))
    features = np.load(str(features_file))

    original_points = np.array(pc.vertices)
    original_colors = np.array(pc.colors)
    print(f"Data loaded. Total points: {len(original_points)}")

    # --- Load Model and Process Query ---
    print(f"Loading CLIP model ({cfg.models.clip_version})...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _ = clip.load(cfg.models.clip_version, device=device)

    text = clip.tokenize([cfg.text_query]).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text)
        text_features /= text_features.norm(dim=-1, keepdim=True)
    query_feature = text_features.cpu().numpy()

    # --- Calculate Similarity and Find Top K ---
    print("Calculating similarity...")
    similarities = cosine_similarity(query_feature, features)[0]
    similar_indices = np.argsort(similarities)[-cfg.top_k:]
    print(f"Top {cfg.top_k} similar point indices found.")

    # --- Create Visualization ---
    validation_colors = np.mean(original_colors[:, :3], axis=1, keepdims=True).astype(np.uint8)
    validation_colors = np.tile(validation_colors, (1, 3))
    # Color the top K similar points green
    validation_colors[similar_indices] = [0, 255, 0]

    # --- Save Output ---
    output_dir = Path(cfg.paths.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    query_slug = cfg.text_query.lower().replace(" ", "_")
    output_file = output_dir / f"clip_highlight_{query_slug}.ply"

    print(f"💾 Saving validation point cloud to {output_file}")
    validation_pc = trimesh.PointCloud(vertices=original_points, colors=validation_colors)
    validation_pc.export(str(output_file))

    print("\n✅ Done! Look for the GREEN points in the new .ply file.")

if __name__ == "__main__":
    highlight_by_text()