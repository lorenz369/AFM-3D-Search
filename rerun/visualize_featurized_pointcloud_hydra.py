#!/usr/bin/env python3
"""
Featurized Pointcloud visualization script using Rerun SDK (Hydra-powered)
Visualizes the featurized point clouds from locate3d preprocessing pipeline
with support for feature visualization through PCA color mapping, surface reconstruction,
and dynamic random highlighting

Usage examples:
  # Basic exploration
  python visualize_featurized_pointcloud_hydra.py pointcloud_dir=/path/to/data

  # Use preset for text search
  python visualize_featurized_pointcloud_hydra.py --config-name=text_search pointcloud_dir=/path/to/data

  # Override specific settings
  python visualize_featurized_pointcloud_hydra.py pointcloud_dir=/path/to/data rendering.create_mesh=true

  # Text search with custom query
  python visualize_featurized_pointcloud_hydra.py --config-name=text_search pointcloud_dir=/path/to/data text_similarity.query="red car"
"""

import rerun as rr
import numpy as np
import open3d as o3d
import torch
import os
import glob
import time
import threading
import sys
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
from dataclasses import dataclass, field
from typing import Optional, List
import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf

# Import all necessary functions from the original script
from visualize_featurized_pointcloud import (
    discover_featurized_files,
    load_featurized_pointcloud,
    features_to_colors_pca,
    estimate_normals_and_mesh,
    create_voxel_grid,
    create_random_highlights,
    animate_highlights,
    create_voxel_highlights,
    create_text_similarity_highlights,
    HAS_CLIP_ENCODER,
    HAS_OPEN3D,
    HAS_SCIPY
)

# Add the path to locate-3d for importing ClipEncoder
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'locate-3d'))
try:
    from preprocessing.image_features.clip_encoder import ClipEncoder
except ImportError:
    print("Warning: ClipEncoder not available. Text-based similarity highlighting will be disabled.")

# ==================== Configuration Classes ====================

@dataclass
class ServerConfig:
    port: int = 9878

@dataclass
class RenderingConfig:
    point_size: float = 0.01
    create_mesh: bool = False
    mesh_method: str = "poisson"
    alpha_value: float = 0.03  # Added alpha value parameter
    use_voxels: bool = False
    voxel_size: float = 0.05

@dataclass
class FeaturesConfig:
    show_features: bool = True
    feature_type: str = "both"  # clip, dino, both
    pca_method: str = "hsv"     # rgb, hsv

@dataclass
class HighlightingConfig:
    enable: bool = False
    mode: str = "random_points"
    animate: bool = False
    animation_duration: float = 30.0

@dataclass
class TextSimilarityConfig:
    query: Optional[str] = None
    top_k: int = 100
    threshold: float = 0.3
    clip_model_version: str = "ViT-B/32"

@dataclass
class InteractiveSearchConfig:
    enable: bool = False
    outlier_method: str = "adaptive"  # iqr, percentile, z_score, adaptive, combined
    use_statistical_outliers: bool = True
    use_dino_filtering: bool = True

@dataclass
class VisualizationConfig:
    # Basic settings
    original_pointcloud_path: Optional[str] = None
    pointcloud_dir: str = "???"
    file_type: str = "combined"
    mode: str = "serve"
    
    # Component configs - use default_factory for mutable defaults
    server: ServerConfig = field(default_factory=ServerConfig)
    rendering: RenderingConfig = field(default_factory=RenderingConfig)
    features: FeaturesConfig = field(default_factory=FeaturesConfig)
    highlighting: HighlightingConfig = field(default_factory=HighlightingConfig)
    text_similarity: TextSimilarityConfig = field(default_factory=TextSimilarityConfig)
    interactive_search: InteractiveSearchConfig = field(default_factory=InteractiveSearchConfig)

# Register config classes with Hydra
cs = ConfigStore.instance()
cs.store(name="base_config", node=VisualizationConfig)

# ==================== Hydra-powered Visualization Function ====================

def visualize_featurized_pointcloud_hydra(cfg: DictConfig, files_info: dict):
    """Enhanced visualization function that uses Hydra config."""
    
    print(f"=== Enhanced Visualization with Hydra Config ===")
    
    # Initialize CLIP encoder if text query is provided
    clip_encoder = None
    if cfg.text_similarity.query and HAS_CLIP_ENCODER:
        # Auto-detect CLIP model version based on features
        try:
            print("🔍 Checking point cloud feature dimensions...")
            temp_points, temp_rgb, temp_features_info = load_featurized_pointcloud(files_info[cfg.file_type])
            
            if 'clip' in temp_features_info:
                feature_dim = temp_features_info['clip'].shape[1]
                if feature_dim == 512:
                    detected_clip_version = "ViT-B/32"
                elif feature_dim == 768:
                    detected_clip_version = "ViT-L/14"
                else:
                    detected_clip_version = cfg.text_similarity.clip_model_version
                
                print(f"🤖 Initializing CLIP encoder ({detected_clip_version})...")
                clip_encoder = ClipEncoder(version=detected_clip_version)
                print(f"✓ CLIP encoder ready")
            
        except Exception as e:
            print(f"❌ Failed to initialize CLIP encoder: {e}")
    
    # Initialize Rerun
    rr.init("Enhanced_Featurized_Pointcloud_Hydra", spawn=False)
    
    if cfg.mode == "save":
        rr.save("featurized_output.rrd")
    elif cfg.mode == "serve":
        rr.serve_grpc(grpc_port=cfg.server.port)
        print(f"🖥️  Local gRPC on port {cfg.server.port}")
    elif cfg.mode == "web":
        rr.serve_web(open_browser=True, web_port=9090)
        print("🌐 Web mode: localhost:9090")
    
    # Set up coordinate frame
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Y_UP, static=True)
    rr.set_time_seconds("timeline", 0.0)
    
    # Load pointcloud
    # If the original pointcloud path is available in the config, log the original dense pointcloud for reference
    if hasattr(cfg, "original_pointcloud_path") and cfg.original_pointcloud_path:
        print(f"🔍 Logging original pointcloud from {cfg.original_pointcloud_path}...")
        pcd = o3d.io.read_point_cloud(cfg.original_pointcloud_path)
        points_orig = np.asarray(pcd.points)
        if pcd.has_colors():
            colors_orig = np.asarray(pcd.colors)
        else:
            colors_orig = np.ones_like(points_orig) * 0.8  # Gray fallback
        rr.log("world/dense_original_pointcloud", rr.Points3D(positions=points_orig, colors=colors_orig, radii=0.01))

    # Add featurized pointcloud
    points, rgb, features_info = load_featurized_pointcloud(files_info[cfg.file_type])
    
    # Apply voxel grid if requested
    if cfg.rendering.use_voxels:
        points, rgb = create_voxel_grid(points, rgb, cfg.rendering.voxel_size)
    
    # Log voxelized RGB pointcloud
    rr.log("world/voxelized_pointcloud_rgb", 
           rr.Points3D(points, colors=rgb, radii=cfg.rendering.point_size), 
           static=True)
    
    # Create mesh if requested
    vertices, faces, vertex_colors = None, None, None
    if cfg.rendering.create_mesh and HAS_OPEN3D:
        print("🔺 Creating surface mesh...")
        vertices, faces, vertex_colors = estimate_normals_and_mesh(
            points, rgb, 
            method=cfg.rendering.mesh_method,
            alpha_value=cfg.rendering.alpha_value  # Pass alpha value to the function
        )
        
        if vertices is not None and faces is not None:
            if vertex_colors is not None:
                rr.log("world/mesh_rgb", 
                       rr.Mesh3D(vertex_positions=vertices, 
                               triangle_indices=faces,
                               vertex_colors=vertex_colors), 
                       static=True)
            else:
                rr.log("world/mesh_rgb", 
                       rr.Mesh3D(vertex_positions=vertices, 
                               triangle_indices=faces), 
                       static=True)
            print(f"✓ Surface mesh logged ({len(vertices)} vertices, {len(faces)} faces)")
    
    # Feature visualizations
    if cfg.features.show_features and features_info:
        if 'clip' in features_info and cfg.features.feature_type in ['clip', 'both']:
            clip_colors = features_to_colors_pca(features_info['clip'], method=cfg.features.pca_method)
            rr.log("world/pointcloud_clip_features", 
                   rr.Points3D(points, colors=clip_colors, radii=cfg.rendering.point_size), 
                   static=True)
        
        if 'dino' in features_info and cfg.features.feature_type in ['dino', 'both']:
            dino_colors = features_to_colors_pca(features_info['dino'], method=cfg.features.pca_method)
            rr.log("world/pointcloud_dino_features", 
                   rr.Points3D(points, colors=dino_colors, radii=cfg.rendering.point_size), 
                   static=True)
    
    # Text-based highlighting
    if cfg.highlighting.enable and cfg.text_similarity.query and clip_encoder and 'clip' in features_info:
        highlight_indices, highlight_points, highlight_colors, similarities = create_text_similarity_highlights(
            points, features_info['clip'], cfg.text_similarity.query,
            clip_encoder=clip_encoder,
            top_k=cfg.text_similarity.top_k,
            similarity_threshold=cfg.text_similarity.threshold,
            highlight_color=[1.0, 0.8, 0.0]
        )
        
        if len(highlight_points) > 0:
            rr.log("world/text_similarity_highlights", 
                   rr.Points3D(highlight_points, colors=highlight_colors, radii=0.03))
            print(f"✨ Text similarity highlights: {len(highlight_points)} points")
    
    # Voxel-based highlighting
    elif cfg.highlighting.enable and cfg.highlighting.mode == 'voxel_highlights':
        highlighted_points, highlight_colors, voxel_centers = create_voxel_highlights(
            points, rgb, voxel_size=0.1, highlight_ratio=0.1
        )
        
        # Log highlighted points
        rr.log("world/voxel_highlights", 
               rr.Points3D(highlighted_points, colors=highlight_colors, radii=0.02), 
               static=True)
        
        # Log voxel centers as larger cubes
        rr.log("world/voxel_centers", 
               rr.Points3D(voxel_centers, 
                         colors=[[1.0, 1.0, 0.0]] * len(voxel_centers), 
                         radii=0.05), 
               static=True)
    
    # Standard highlighting
    elif cfg.highlighting.enable:
        highlight_indices, highlight_points, highlight_colors = create_random_highlights(
            points, rgb, highlight_mode=cfg.highlighting.mode, num_highlights=200
        )
        rr.log("world/static_highlights", 
               rr.Points3D(highlight_points, colors=highlight_colors, radii=0.025), 
               static=True)
    
    # Animation
    if cfg.highlighting.animate:
        animate_highlights(
            points, rgb, "world/pointcloud_rgb", 
            duration=cfg.highlighting.animation_duration,
            highlight_interval=0.5,
            num_highlights=100
        )
    
    # Log comprehensive statistics with better structure
    print("📊 Logging pointcloud statistics...")
    
    # Basic point cloud info
    bbox_min = points.min(axis=0)
    bbox_max = points.max(axis=0)
    bbox_size = bbox_max - bbox_min
    
    # Log individual scalar values for pointcloud statistics
    # rr.log("stats/num_points", rr.Scalar(len(points)), static=True)  # Total number of points
    # rr.log("stats/bbox_size_x", rr.Scalar(float(bbox_size[0])), static=True)  # Bounding box size in X
    # rr.log("stats/bbox_size_y", rr.Scalar(float(bbox_size[1])), static=True)  # Bounding box size in Y
    # rr.log("stats/bbox_size_z", rr.Scalar(float(bbox_size[2])), static=True)  # Bounding box size in Z
    # rr.log("stats/bbox_volume", rr.Scalar(float(np.prod(bbox_size))), static=True)  # Bounding box volume
    
    # Summary text log with all key information
    summary_text = f"""📊 POINTCLOUD SUMMARY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🔢 Points: {len(points):,}
📦 Bounding Box:
   • Min: [{bbox_min[0]:.3f}, {bbox_min[1]:.3f}, {bbox_min[2]:.3f}]
   • Max: [{bbox_max[0]:.3f}, {bbox_max[1]:.3f}, {bbox_max[2]:.3f}]
   • Size: [{bbox_size[0]:.3f}, {bbox_size[1]:.3f}, {bbox_size[2]:.3f}]
   • Volume: {np.prod(bbox_size):.3f} cubic units
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"""
    
    # Add feature information if available
    if features_info:
        summary_text += "\n🧠 Features:\n"
        for feat_name, features in features_info.items():
            feat_dim = features.shape[1]
            feat_mean_norm = np.linalg.norm(features, axis=1).mean()
            feat_std_norm = np.linalg.norm(features, axis=1).std()
            
            # Log individual feature stats
            #rr.log(f"stats/features_{feat_name}_dim", rr.Scalar(feat_dim), static=True)
            #rr.log(f"stats/features_{feat_name}_mean_norm", rr.Scalar(float(feat_mean_norm)), static=True)
            #rr.log(f"stats/features_{feat_name}_std_norm", rr.Scalar(float(feat_std_norm)), static=True)
            
            summary_text += f"   • {feat_name.upper()}: {feat_dim}D features, norm μ={feat_mean_norm:.3f} σ={feat_std_norm:.3f}\n"
    
    summary_text += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    # Log the comprehensive summary as a text log
    #rr.log("stats/summary", rr.TextLog(summary_text, level=rr.TextLogLevel.INFO), static=True)
    
    # Also print to console for immediate reference
    print(summary_text)
    
    print("🎉 Visualization complete!")
    
    if cfg.mode == "serve":
        print("Press Enter to exit the server.")
        input()

@hydra.main(version_base=None, config_path="config", config_name="base_config")
def main(cfg: DictConfig) -> None:
    """Main function using Hydra configuration."""
    
    print("🚀 Starting Enhanced Featurized Pointcloud Visualization (Hydra-powered)")
    print(f"📋 Configuration:")
    print(OmegaConf.to_yaml(cfg, resolve=True))
    
    # Auto-discover featurized pointcloud files
    try:
        files_info = discover_featurized_files(cfg.pointcloud_dir)
    except Exception as e:
        print(f"Error discovering files: {e}")
        return

    # Validate file type selection
    if cfg.file_type not in files_info:
        print(f"Error: File type '{cfg.file_type}' not found.")
        print(f"Available file types: {list(files_info.keys())}")
        return

    # Check if interactive mode is enabled
    if cfg.interactive_search.enable:
        print("⚡ INTERACTIVE MODE ENABLED ⚡")

        # Lazily import to avoid issues if dependencies are not installed
        from visualize_interactive_text_search import InteractiveTextSearch, HAS_CLIP_ENCODER as HAS_INTERACTIVE_CLIP_ENCODER, HAS_OPEN3D as HAS_INTERACTIVE_OPEN3D

        if not HAS_INTERACTIVE_CLIP_ENCODER:
            print("❌ ClipEncoder not available. Interactive search requires CLIP functionality.")
            return

        create_mesh_flag = cfg.rendering.create_mesh
        if create_mesh_flag and not HAS_INTERACTIVE_OPEN3D:
             print("⚠️  Open3D not available. Disabling mesh creation for interactive mode.")
             create_mesh_flag = False

        try:
            interactive_search = InteractiveTextSearch(
                files_info=files_info,
                file_key=cfg.file_type,
                clip_model_version=cfg.text_similarity.clip_model_version,
                create_mesh=create_mesh_flag,
                outlier_method=cfg.interactive_search.outlier_method,
                use_statistical_outliers=cfg.interactive_search.use_statistical_outliers,
                use_dino_filtering=cfg.interactive_search.use_dino_filtering,
                original_pointcloud_path=cfg.original_pointcloud_path,  # Add this line
            )
            interactive_search.run_interactive_session(port=cfg.server.port)
        except Exception as e:
            print(f"❌ Error starting interactive session: {e}")
            raise

    else:
        # Convert DictConfig to parameters for the visualization function
        visualize_featurized_pointcloud_hydra(cfg, files_info)

if __name__ == "__main__":
    main() 