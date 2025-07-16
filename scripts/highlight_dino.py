# src/afm_3d_search/highlight_dino.py

import hydra
from omegaconf import DictConfig
import numpy as np
import trimesh
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path

@hydra.main(config_path="conf", config_name="highlight_dino", version_base=None)
def highlight_by_similarity(cfg: DictConfig) -> None:
    """
    Loads a point cloud and its DINO features, and highlights points
    that are similar to a given query point index.
    """
    print(f"🔎 Starting DINO highlight for point index: {cfg.query_point_index}")
    
    run_dir = Path(hydra.utils.to_absolute_path(cfg.paths.run_output_dir))
    base_name = run_dir.name

    ply_file = run_dir / f"{base_name}.ply"
    features_file = run_dir / f"{base_name}_dino_features.npy"

    # --- Load Data ---
    if not ply_file.exists() or not features_file.exists():
        print(f"❌ Error: Input files not found in {run_dir}")
        print("Please run the main 'generate_pointcloud.py' script first.")
        return
        
    print("Loading data...")
    pc = trimesh.load(str(ply_file))
    features = np.load(str(features_file))
    
    original_points = np.array(pc.vertices)
    original_colors = np.array(pc.colors)
    print(f"Data loaded. Total points: {len(original_points)}")

    if cfg.query_point_index >= len(original_points):
        print(f"❌ Error: query_point_index ({cfg.query_point_index}) is out of bounds.")
        return

    # --- Select Query and Calculate Similarity ---
    query_feature = features[cfg.query_point_index].reshape(1, -1)
    
    print(f"Calculating similarity for point {cfg.query_point_index}...")
    similarities = cosine_similarity(query_feature, features)[0]
    similar_indices = np.argsort(similarities)[- (cfg.top_k + 1):]
    print(f"Top {cfg.top_k} similar point indices found.")

    # --- Create Visualization ---
    validation_colors = np.mean(original_colors[:, :3], axis=1, keepdims=True).astype(np.uint8)
    validation_colors = np.tile(validation_colors, (1, 3))
    
    # Color the similar points yellow
    validation_colors[similar_indices] = [255, 255, 0] # Yellow
    # Color the original query point red
    validation_colors[cfg.query_point_index] = [255, 0, 0] # Red

    # --- Save Output ---
    output_dir = Path(cfg.paths.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"dino_highlight_{cfg.query_point_index}.ply"
    
    print(f"💾 Saving validation point cloud to {output_file}")
    validation_pc = trimesh.PointCloud(vertices=original_points, colors=validation_colors)
    validation_pc.export(str(output_file))

    print("\n✅ Done! Look for the RED and YELLOW points in the new .ply file.")


if __name__ == "__main__":
    highlight_by_similarity()