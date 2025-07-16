import hydra
import numpy as np
import trimesh
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path

from conf.schema import MainConfig
import hydra.core.hydra_config

@hydra.main(config_path="conf", config_name="config", version_base=None)
def highlight_by_similarity(cfg: MainConfig) -> None:
    """
    Loads a DINO feature point cloud and highlights points
    that are similar to a given query point index.
    """
    print(f"🔎 Starting DINO highlight for scene '{cfg.scene_id}'")
    print(f"Querying with point index: {cfg.highlight.query_point_index}")

    # --- Load Data from a completed run ---
    # Construct the path to the pipeline's output directory using the scene_id
    input_dir = Path(cfg.paths.data_root) / cfg.paths.completed_dir_name / cfg.scene_id
    
    ply_file = input_dir / cfg.paths.ply_filename
    features_file = input_dir / cfg.paths.dino_features_filename
    
    if not ply_file.exists() or not features_file.exists():
        print(f"❌ Error: Input files not found in {input_dir}")
        print(f"Please run the main pipeline for scene_id='{cfg.scene_id}' first.")
        return
        
    print("Loading point cloud and DINO features...")
    pc = trimesh.load(str(ply_file))
    features = np.load(str(features_file))
    
    original_points = np.array(pc.vertices)
    original_colors = np.array(pc.colors)
    print(f"Data loaded. Total points: {len(original_points)}")

    if cfg.highlight.query_point_index >= len(original_points):
        print(f"❌ Error: query_point_index ({cfg.highlight.query_point_index}) is out of bounds for {len(original_points)} points.")
        return

    # --- Select Query and Calculate Similarity ---
    query_feature = features[cfg.highlight.query_point_index].reshape(1, -1)
    
    print(f"Calculating similarity for point {cfg.highlight.query_point_index}...")
    similarities = cosine_similarity(query_feature, features)[0]
    similar_indices = np.argsort(similarities)[- (cfg.highlight.top_k + 1):] # +1 to include the query point itself
    print(f"Top {cfg.highlight.top_k} similar point indices found.")

    # --- Create Visualization ---
    validation_colors = np.mean(original_colors[:, :3], axis=1, keepdims=True).astype(np.uint8)
    validation_colors = np.tile(validation_colors, (1, 3))
    
    # Color the similar points yellow
    validation_colors[similar_indices] = [255, 255, 0] # Yellow
    # Color the original query point red for easy identification
    validation_colors[cfg.highlight.query_point_index] = [255, 0, 0] # Red

    # --- Save Output ---
    # The output of this highlight script will go into the standard `outputs/`
    output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    output_file = output_dir / f"dino_highlight_{cfg.highlight.query_point_index}.ply"
    
    print(f"💾 Saving validation point cloud to {output_file}")
    validation_pc = trimesh.PointCloud(vertices=original_points, colors=validation_colors)
    validation_pc.export(str(output_file))

    print("\n✅ Done! Check the new timestamped folder inside 'outputs' for the result.")


if __name__ == "__main__":
    highlight_by_similarity()