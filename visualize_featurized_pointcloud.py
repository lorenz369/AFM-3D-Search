#!/usr/bin/env python3
"""
Featurized Pointcloud visualization script using Rerun SDK
Visualizes the featurized point clouds from MASt3R preprocessing pipeline
with support for feature visualization through PCA color mapping
"""

import rerun as rr
import numpy as np
import torch
import argparse
import os
import glob
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb

def discover_featurized_files(base_dir):
    """Auto-discover featurized pointcloud files (.pt)s in the directory."""
    base_dir = os.path.abspath(base_dir)
    
    # Find all .pt files
    pt_files = glob.glob(os.path.join(base_dir, "*.pt"))
    if not pt_files:
        raise FileNotFoundError(f"No .pt files found in {base_dir}")
    
    files_info = {}
    for pt_file in pt_files:
        filename = os.path.basename(pt_file)
        if 'clip' in filename.lower():
            files_info['clip'] = pt_file
        elif 'dino' in filename.lower():
            files_info['dino'] = pt_file
        elif 'combined' in filename.lower():
            files_info['combined'] = pt_file
        else:
            # Generic naming
            base_name = os.path.splitext(filename)[0]
            files_info[base_name] = pt_file
    
    print(f"Discovered featurized pointcloud files in {base_dir}:")
    for key, path in files_info.items():
        file_size = os.path.getsize(path) / (1024**3)  # GB
        print(f"  {key}: {os.path.basename(path)} ({file_size:.1f} GB)")
    
    return files_info

def load_featurized_pointcloud(pt_path):
    """Load featurized pointcloud from .pt file."""
    print(f"Loading featurized pointcloud from {pt_path}...")
    
    try:
        data = torch.load(pt_path, map_location='cpu')
        
        # Extract data
        points = data['points'].numpy() if isinstance(data['points'], torch.Tensor) else data['points']
        rgb = data['rgb'].numpy() if isinstance(data['rgb'], torch.Tensor) else data['rgb']
        
        # Ensure RGB is in [0,1] range
        if rgb.max() > 1.0:
            rgb = rgb / 255.0
        
        features_info = {}
        if 'features_clip' in data:
            features_clip = data['features_clip'].numpy() if isinstance(data['features_clip'], torch.Tensor) else data['features_clip']
            features_info['clip'] = features_clip
            print(f"  CLIP features: {features_clip.shape}")
        
        if 'features_dino' in data:
            features_dino = data['features_dino'].numpy() if isinstance(data['features_dino'], torch.Tensor) else data['features_dino']
            features_info['dino'] = features_dino
            print(f"  DINO features: {features_dino.shape}")
        
        print(f"Loaded pointcloud: {points.shape[0]} points, RGB shape: {rgb.shape}")
        
        return points, rgb, features_info
        
    except Exception as e:
        print(f"Error loading {pt_path}: {e}")
        raise

def features_to_colors_pca(features, n_components=3, method='hsv'):
    """
    Convert high-dimensional features to RGB colors using PCA.
    
    Args:
        features: numpy array of shape [N, D] 
        n_components: number of PCA components (usually 3 for RGB)
        method: 'rgb' for direct mapping, 'hsv' for HSV-based coloring
    """
    print(f"Converting {features.shape[1]}D features to colors using PCA...")
    
    # Standardize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Apply PCA
    pca = PCA(n_components=n_components)
    features_pca = pca.fit_transform(features_scaled)
    
    print(f"PCA explained variance ratio: {pca.explained_variance_ratio_}")
    
    if method == 'rgb':
        # Direct RGB mapping - normalize to [0,1]
        colors = features_pca.copy()
        for i in range(n_components):
            col = colors[:, i]
            colors[:, i] = (col - col.min()) / (col.max() - col.min())
        
        # Pad with zeros if needed or truncate
        if colors.shape[1] < 3:
            colors = np.pad(colors, ((0, 0), (0, 3 - colors.shape[1])), mode='constant')
        else:
            colors = colors[:, :3]
            
    elif method == 'hsv':
        # HSV-based coloring for better visual separation
        if n_components >= 2:
            # Use first two components for hue and saturation
            hue = (features_pca[:, 0] - features_pca[:, 0].min()) / (features_pca[:, 0].max() - features_pca[:, 0].min())
            sat = (features_pca[:, 1] - features_pca[:, 1].min()) / (features_pca[:, 1].max() - features_pca[:, 1].min())
            
            # Use third component for value if available, otherwise use constant
            if n_components >= 3:
                val = (features_pca[:, 2] - features_pca[:, 2].min()) / (features_pca[:, 2].max() - features_pca[:, 2].min())
                val = 0.5 + 0.5 * val  # Keep values bright
            else:
                val = np.ones_like(hue) * 0.8
            
            # Convert HSV to RGB
            hsv = np.stack([hue, sat, val], axis=-1)
            colors = hsv_to_rgb(hsv)
        else:
            # Fallback to grayscale
            gray = (features_pca[:, 0] - features_pca[:, 0].min()) / (features_pca[:, 0].max() - features_pca[:, 0].min())
            colors = np.stack([gray, gray, gray], axis=-1)
    
    return colors

def visualize_featurized_pointcloud(
    files_info,
    file_key='combined',
    mode="serve",
    remote_host=None, 
    remote_port=9876,
    show_features=True,
    feature_type='both',
    pca_method='hsv'
):
    """Visualize featurized pointcloud data."""
    
    print(f"=== Starting Featurized Pointcloud Visualization ===")
    print(f"Mode: {mode}")
    print(f"Remote host: {remote_host}")
    print(f"Remote port: {remote_port}")
    print(f"Show features: {show_features}")
    print(f"Feature type: {feature_type}")
    
    if file_key not in files_info:
        print(f"Error: File key '{file_key}' not found. Available: {list(files_info.keys())}")
        return
    
    rr.init("Featurized_Pointcloud_Visualization", spawn=False)
    print("✓ Rerun SDK initialized")

    if mode == "save":
        print("📁 Using SAVE mode - will write to featurized_output.rrd")
        rr.save("featurized_output.rrd")
    elif mode == "serve":
        print("🌐 Using SERVE mode")
        if remote_host:
            print(f"🔗 Remote mode: connecting to gRPC at {remote_host}:{remote_port}")
            rr.serve_grpc(grpc_port=remote_port)
        else:
            print("🖥️  Local mode: spawning Rerun viewer window")
            rr.spawn()
    else:
        print(f"❌ Unknown mode: {mode}")
        
    print("✓ Viewer setup complete")

    # Set up coordinate frame
    print("🌍 Setting up world coordinate frame...")
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Y_UP, static=True)

    # Load the featurized pointcloud
    print("☁️  Loading featurized pointcloud...")
    points, rgb, features_info = load_featurized_pointcloud(files_info[file_key])

    # Log the original RGB pointcloud
    print(f"✓ Logging {len(points)} points with original RGB colors")
    rr.log("world/pointcloud_rgb", rr.Points3D(points, colors=rgb), static=True)

    if show_features and features_info:
        print("🎨 Generating feature-based visualizations...")
        
        # Visualize CLIP features if available and requested
        if 'clip' in features_info and feature_type in ['clip', 'both']:
            print("🔍 Visualizing CLIP features...")
            clip_colors = features_to_colors_pca(features_info['clip'], method=pca_method)
            rr.log("world/pointcloud_clip_features", rr.Points3D(points, colors=clip_colors), static=True)
            print(f"✓ CLIP feature visualization complete ({features_info['clip'].shape[1]}D → RGB)")
        
        # Visualize DINO features if available and requested  
        if 'dino' in features_info and feature_type in ['dino', 'both']:
            print("🦕 Visualizing DINO features...")
            dino_colors = features_to_colors_pca(features_info['dino'], method=pca_method)
            rr.log("world/pointcloud_dino_features", rr.Points3D(points, colors=dino_colors), static=True)
            print(f"✓ DINO feature visualization complete ({features_info['dino'].shape[1]}D → RGB)")

    # Log some basic statistics
    print("📊 Logging pointcloud statistics...")
    rr.log("stats/num_points", rr.Scalar(len(points)), static=True)
    rr.log("stats/bbox_min", rr.Scalar(points.min(axis=0)), static=True)
    rr.log("stats/bbox_max", rr.Scalar(points.max(axis=0)), static=True)
    
    if features_info:
        for feat_name, features in features_info.items():
            rr.log(f"stats/features_{feat_name}_dim", rr.Scalar(features.shape[1]), static=True)
            rr.log(f"stats/features_{feat_name}_mean_norm", rr.Scalar(np.linalg.norm(features, axis=1).mean()), static=True)

    print("🎉 Featurized pointcloud visualization complete!")
    print("\n📋 Available views in Rerun:")
    print("  - world/pointcloud_rgb: Original RGB colors")
    if show_features and features_info:
        if 'clip' in features_info and feature_type in ['clip', 'both']:
            print("  - world/pointcloud_clip_features: CLIP features as colors")
        if 'dino' in features_info and feature_type in ['dino', 'both']:
            print("  - world/pointcloud_dino_features: DINO features as colors")
    print("  - stats/*: Various statistics")
    
    if remote_host:
        print(f"🌐 Streaming to {remote_host}:{remote_port}. Open the Rerun viewer and connect.")
    else:
        print("🖥️  Check the Rerun viewer window that should have opened.")

def main():
    parser = argparse.ArgumentParser(description="Visualize featurized pointclouds from MASt3R preprocessing")
    parser.add_argument("pointcloud_dir", 
                       help="Directory containing featurized pointcloud .pt files")
    parser.add_argument("--file-type",
                       choices=["clip", "dino", "combined"],
                       default="combined",
                       help="Which .pt file to visualize")
    parser.add_argument("--mode",
                       choices=["serve", "save"],
                       default="serve",
                       help="Set the visualization mode: 'serve' to stream, 'save' to file.")
    parser.add_argument("--remote-host",
                       help="Remote host IP for Rerun streaming")
    parser.add_argument("--remote-port", type=int, default=9876)
    parser.add_argument("--no-features", action="store_true",
                       help="Skip feature visualization (only show RGB)")
    parser.add_argument("--feature-type",
                       choices=["clip", "dino", "both"],
                       default="both",
                       help="Which features to visualize")
    parser.add_argument("--pca-method",
                       choices=["rgb", "hsv"],
                       default="hsv",
                       help="Method for converting features to colors")
    
    args = parser.parse_args()

    # Auto-discover featurized pointcloud files
    try:
        files_info = discover_featurized_files(args.pointcloud_dir)
    except Exception as e:
        print(f"Error discovering files: {e}")
        return

    visualize_featurized_pointcloud(
        files_info,
        file_key=args.file_type,
        mode=args.mode,
        remote_host=args.remote_host, 
        remote_port=args.remote_port,
        show_features=not args.no_features,
        feature_type=args.feature_type,
        pca_method=args.pca_method
    )

    if args.mode == "serve" and args.remote_host:
        print("\nVisualization server is running. Press Enter on the server terminal to exit.")
        input()

if __name__ == "__main__":
    main() 