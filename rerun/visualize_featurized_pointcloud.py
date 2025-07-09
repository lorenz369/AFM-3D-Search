#!/usr/bin/env python3
"""
Featurized Pointcloud visualization script using Rerun SDK
Visualizes the featurized point clouds from MASt3R preprocessing pipeline
with support for feature visualization through PCA color mapping, surface reconstruction,
and dynamic random highlighting
"""

import rerun as rr
import numpy as np
import torch
import argparse
import os
import glob
import time
import threading
import sys
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb

# Add the path to locate-3d for importing ClipEncoder
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'locate-3d'))
try:
    from preprocessing.image_features.clip_encoder import ClipEncoder
    HAS_CLIP_ENCODER = True
except ImportError:
    HAS_CLIP_ENCODER = False
    print("Warning: ClipEncoder not available. Text-based similarity highlighting will be disabled.")

# Try importing surface reconstruction libraries
try:
    import open3d as o3d
    HAS_OPEN3D = True
except ImportError:
    HAS_OPEN3D = False
    print("Warning: Open3D not available. Surface reconstruction features will be disabled.")

try:
    from scipy.spatial import Delaunay
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("Warning: SciPy not available. Some interpolation features will be disabled.")

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
            if features_clip is not None:
                features_info['clip'] = features_clip
                print(f"  CLIP features: {features_clip.shape}")
            else:
                print("  CLIP features: None")
        
        if 'features_dino' in data:
            features_dino = data['features_dino'].numpy() if isinstance(data['features_dino'], torch.Tensor) else data['features_dino']
            if features_dino is not None:
                features_info['dino'] = features_dino
                print(f"  DINO features: {features_dino.shape}")
            else:
                print("  DINO features: None")
        
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

def estimate_normals_and_mesh(points, colors=None, method='poisson', depth=9, density_threshold=0.1, alpha_value=0.03):
    """
    Create surface mesh from pointcloud using various reconstruction methods.
    
    Args:
        points: numpy array [N, 3]
        colors: numpy array [N, 3] optional
        method: 'poisson', 'ball_pivoting', or 'alpha_shape'
        depth: depth for Poisson reconstruction
        density_threshold: threshold for removing low-density vertices
        alpha_value: alpha value for alpha shape reconstruction (smaller = denser mesh)
    
    Returns:
        vertices, faces, vertex_colors (if colors provided)
    """
    if not HAS_OPEN3D:
        print("Open3D not available - skipping mesh reconstruction")
        return None, None, None
    
    print(f"Creating surface mesh using {method} reconstruction...")
    
    # Create Open3D pointcloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # Estimate normals
    print("Estimating surface normals...")
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    pcd.orient_normals_consistent_tangent_plane(k=15)
    
    if method == 'poisson':
        print(f"Running Poisson surface reconstruction (depth={depth})...")
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=depth, width=0, scale=1.1, linear_fit=False
        )
        
        # Remove low-density vertices
        if density_threshold > 0:
            print(f"Removing vertices with density < {density_threshold}...")
            vertices_to_remove = densities < np.quantile(densities, density_threshold)
            mesh.remove_vertices_by_mask(vertices_to_remove)
    
    elif method == 'ball_pivoting':
        print("Running Ball Pivoting reconstruction...")
        # Estimate radius for ball pivoting
        distances = pcd.compute_nearest_neighbor_distance()
        avg_dist = np.mean(distances)
        radius = 1.5 * avg_dist
        radii = [radius, radius * 2, radius * 4]
        
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
            pcd, o3d.utility.DoubleVector(radii)
        )
    
    elif method == 'alpha_shape':
        print(f"Running Alpha Shape reconstruction (alpha={alpha_value})...")
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha_value)
    
    else:
        print(f"Unknown reconstruction method: {method}")
        return None, None, None
    
    # Extract vertices and faces
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)
    
    # Extract vertex colors if available
    vertex_colors = None
    if colors is not None and len(mesh.vertex_colors) > 0:
        vertex_colors = np.asarray(mesh.vertex_colors)
    elif colors is not None:
        # Interpolate colors to mesh vertices
        print("Interpolating colors to mesh vertices...")
        from scipy.spatial import cKDTree
        tree = cKDTree(points)
        distances, indices = tree.query(vertices, k=3)
        
        # Inverse distance weighting
        weights = 1.0 / (distances + 1e-8)
        weights = weights / weights.sum(axis=1, keepdims=True)
        vertex_colors = (weights[:, :, np.newaxis] * colors[indices]).sum(axis=1)
    
    print(f"Generated mesh: {len(vertices)} vertices, {len(faces)} faces")
    return vertices, faces, vertex_colors

def create_voxel_grid(points, colors=None, voxel_size=0.05):
    """
    Create a voxel grid representation for smoother visualization.
    """
    if not HAS_OPEN3D:
        return points, colors
    
    print(f"Creating voxel grid with size {voxel_size}...")
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # Downsample to voxel grid
    voxel_grid = pcd.voxel_down_sample(voxel_size)
    
    voxel_points = np.asarray(voxel_grid.points)
    voxel_colors = np.asarray(voxel_grid.colors) if colors is not None else None
    
    print(f"Voxel grid: {len(points)} → {len(voxel_points)} points")
    return voxel_points, voxel_colors

def create_random_highlights(points, colors, highlight_mode='random_points', 
                           num_highlights=100, highlight_size=0.05, 
                           highlight_color=[1.0, 0.0, 0.0]):
    """
    Create random highlights for dynamic visualization.
    
    Args:
        points: numpy array [N, 3]
        colors: numpy array [N, 3] 
        highlight_mode: 'random_points', 'spatial_clusters', 'feature_based'
        num_highlights: number of points to highlight
        highlight_size: size of highlighted points
        highlight_color: RGB color for highlights
    
    Returns:
        highlight_indices, highlight_points, highlight_colors
    """
    
    if highlight_mode == 'random_points':
        # Simple random point selection
        highlight_indices = np.random.choice(len(points), 
                                           min(num_highlights, len(points)), 
                                           replace=False)
    
    elif highlight_mode == 'spatial_clusters':
        # Highlight points in spatial clusters - FIXED VERSION
        center_idx = np.random.choice(len(points))
        center_point = points[center_idx]
        
        # Find points within radius of center - use SMALL fixed radius
        distances = np.linalg.norm(points - center_point, axis=1)
        
        # Use much smaller radius for tight clusters
        # Calculate based on point cloud scale
        bbox_size = np.linalg.norm(points.max(axis=0) - points.min(axis=0))
        radius = bbox_size * 0.02  # 2% of bounding box diagonal
        
        nearby_indices = np.where(distances <= radius)[0]
        
        # Take ALL points within radius (no random subsampling)
        # But limit to reasonable number to avoid performance issues
        if len(nearby_indices) > num_highlights:
            # If too many points, take the closest ones
            sorted_indices = nearby_indices[np.argsort(distances[nearby_indices])]
            highlight_indices = sorted_indices[:num_highlights]
        else:
            # Take all nearby points
            highlight_indices = nearby_indices
        
        print(f"Spatial cluster: center at {center_point}, radius {radius:.3f}, {len(highlight_indices)} points")
    
    elif highlight_mode == 'tight_clusters':
        # Create multiple small tight clusters for maximum visibility
        bbox_size = np.linalg.norm(points.max(axis=0) - points.min(axis=0))
        small_radius = bbox_size * 0.01  # Even smaller radius - 1% of bounding box
        num_clusters = 5  # Create 5 small clusters
        points_per_cluster = num_highlights // num_clusters
        
        all_highlight_indices = []
        
        for i in range(num_clusters):
            center_idx = np.random.choice(len(points))
            center_point = points[center_idx]
            distances = np.linalg.norm(points - center_point, axis=1)
            
            nearby_indices = np.where(distances <= small_radius)[0]
            
            if len(nearby_indices) > points_per_cluster:
                # Take closest points
                sorted_indices = nearby_indices[np.argsort(distances[nearby_indices])]
                cluster_indices = sorted_indices[:points_per_cluster]
            else:
                cluster_indices = nearby_indices
            
            all_highlight_indices.extend(cluster_indices)
        
        highlight_indices = np.array(all_highlight_indices)
        print(f"Tight clusters: {num_clusters} clusters, {len(highlight_indices)} total points")
    
    elif highlight_mode == 'feature_based':
        # Highlight points with extreme feature values (if colors represent features)
        if colors is not None:
            # Use color intensity as proxy for "interesting" features
            color_intensity = np.linalg.norm(colors, axis=1)
            # Select both high and low intensity points
            high_intensity_indices = np.argsort(color_intensity)[-num_highlights//2:]
            low_intensity_indices = np.argsort(color_intensity)[:num_highlights//2]
            highlight_indices = np.concatenate([high_intensity_indices, low_intensity_indices])
        else:
            # Fallback to random
            highlight_indices = np.random.choice(len(points), 
                                               min(num_highlights, len(points)), 
                                               replace=False)
    
    else:
        highlight_indices = np.random.choice(len(points), 
                                           min(num_highlights, len(points)), 
                                           replace=False)
    
    highlight_points = points[highlight_indices]
    highlight_colors = np.tile(highlight_color, (len(highlight_indices), 1))
    
    return highlight_indices, highlight_points, highlight_colors

def animate_highlights(points, colors, entity_path, duration=10.0, 
                      highlight_interval=1.0, num_highlights=50, 
                      highlight_modes=['random_points', 'spatial_clusters', 'tight_clusters', 'feature_based']):
    """
    Create animated highlighting effects over time.
    
    Args:
        points: numpy array [N, 3]
        colors: numpy array [N, 3]
        entity_path: rerun entity path
        duration: total animation duration in seconds
        highlight_interval: time between highlight updates
        num_highlights: number of points to highlight at once
        highlight_modes: list of highlighting modes to cycle through
    """
    
    def highlight_thread():
        start_time = time.time()
        mode_idx = 0
        
        print(f"🎬 Starting dynamic highlighting animation for {duration}s...")
        
        while time.time() - start_time < duration:
            current_time = time.time() - start_time
            rr.set_time_seconds("timeline", current_time)
            
            # Cycle through different highlight modes
            current_mode = highlight_modes[mode_idx % len(highlight_modes)]
            
            # Create random highlights
            highlight_indices, highlight_points, highlight_colors = create_random_highlights(
                points, colors, 
                highlight_mode=current_mode,
                num_highlights=num_highlights,
                highlight_color=[1.0, 0.2, 0.2] if current_mode == 'random_points' 
                              else [0.2, 1.0, 0.2] if current_mode == 'spatial_clusters'
                              else [1.0, 1.0, 0.2] if current_mode == 'tight_clusters'
                              else [0.2, 0.2, 1.0]
            )
            
            # Log highlighted points with larger size
            rr.log(f"{entity_path}/highlights", 
                   rr.Points3D(highlight_points, 
                             colors=highlight_colors, 
                             radii=0.03))
            
            # Log highlight info
            rr.log(f"{entity_path}/highlight_info", 
                   rr.TextLog(f"Mode: {current_mode}, Count: {len(highlight_points)}"))
            
            mode_idx += 1
            time.sleep(highlight_interval)
        
        print("✓ Dynamic highlighting animation completed")
    
    # Start animation in separate thread
    thread = threading.Thread(target=highlight_thread, daemon=True)
    thread.start()
    return thread

def create_voxel_highlights(points, colors, voxel_size=0.1, highlight_ratio=0.1):
    """
    Create voxel-based highlighting by selecting random voxels.
    
    Args:
        points: numpy array [N, 3]
        colors: numpy array [N, 3]
        voxel_size: size of voxels
        highlight_ratio: fraction of voxels to highlight
    
    Returns:
        highlighted_points, highlighted_colors, voxel_centers
    """
    
    print(f"🧊 Creating voxel-based highlights (voxel_size={voxel_size})...")
    
    # Create voxel grid
    min_coords = points.min(axis=0)
    max_coords = points.max(axis=0)
    
    # Calculate voxel indices for each point
    voxel_indices = ((points - min_coords) / voxel_size).astype(int)
    
    # Get unique voxels and their point counts
    unique_voxels, inverse_indices, voxel_counts = np.unique(
        voxel_indices, axis=0, return_inverse=True, return_counts=True
    )
    
    # Select random voxels to highlight
    num_voxels_to_highlight = max(1, int(len(unique_voxels) * highlight_ratio))
    highlighted_voxel_indices = np.random.choice(
        len(unique_voxels), num_voxels_to_highlight, replace=False
    )
    
    # Find all points in highlighted voxels
    highlighted_point_mask = np.isin(inverse_indices, highlighted_voxel_indices)
    highlighted_points = points[highlighted_point_mask]
    
    # Create bright highlight colors
    highlight_colors = np.zeros_like(colors[highlighted_point_mask])
    highlight_colors[:, 0] = 1.0  # Bright red
    highlight_colors[:, 1] = 0.5  # Some green
    
    # Calculate voxel centers for visualization
    voxel_centers = min_coords + (unique_voxels[highlighted_voxel_indices] + 0.5) * voxel_size
    
    print(f"✓ Highlighted {len(highlighted_points)} points in {len(highlighted_voxel_indices)} voxels")
    
    return highlighted_points, highlight_colors, voxel_centers

def create_text_similarity_highlights(points, clip_features, text_query, 
                                    clip_encoder=None, top_k=100, 
                                    similarity_threshold=0.3,
                                    highlight_color=[1.0, 0.8, 0.0]):
    """
    Create highlights based on semantic similarity between text query and CLIP features.
    
    Args:
        points: numpy array [N, 3] - 3D points
        clip_features: numpy array [N, D] - CLIP features for each point
        text_query: str - text query to search for
        clip_encoder: ClipEncoder instance
        top_k: int - number of most similar points to highlight
        similarity_threshold: float - minimum cosine similarity threshold
        highlight_color: list - RGB color for highlights
    
    Returns:
        highlight_indices, highlight_points, highlight_colors, similarities
    """
    
    if not HAS_CLIP_ENCODER or clip_encoder is None:
        print("⚠️  CLIP encoder not available for text similarity highlighting")
        return np.array([]), np.array([]).reshape(0, 3), np.array([]).reshape(0, 3), np.array([])
    
    if clip_features is None or len(clip_features) == 0:
        print("⚠️  No CLIP features available for text similarity highlighting")
        return np.array([]), np.array([]).reshape(0, 3), np.array([]).reshape(0, 3), np.array([])
    
    print(f"🔍 Computing text similarity for query: '{text_query}'")
    print(f"📊 CLIP features shape: {clip_features.shape}")
    
    # Encode the text query
    try:
        text_features = clip_encoder.encode_text(text_query)
        text_features = text_features.cpu().numpy()
        
        # Normalize text features
        text_features = text_features / np.linalg.norm(text_features, axis=1, keepdims=True)
        print(f"✓ Text encoded to {text_features.shape}")
        
    except Exception as e:
        print(f"❌ Error encoding text query: {e}")
        return np.array([]), np.array([]).reshape(0, 3), np.array([]).reshape(0, 3), np.array([])
    
    # Check dimension compatibility
    clip_feature_dim = clip_features.shape[1]
    text_feature_dim = text_features.shape[1]
    
    if clip_feature_dim != text_feature_dim:
        print(f"❌ Dimension mismatch: CLIP features ({clip_feature_dim}D) vs Text features ({text_feature_dim}D)")
        print(f"   Point cloud likely created with different CLIP model")
        print(f"   Expected: ViT-L/14 for 768D or ViT-B/32 for 512D")
        return np.array([]), np.array([]).reshape(0, 3), np.array([]).reshape(0, 3), np.array([])
    
    # Normalize CLIP features for cosine similarity
    # Handle potential zero vectors
    norms = np.linalg.norm(clip_features, axis=1, keepdims=True)
    zero_mask = norms[:, 0] == 0
    if np.any(zero_mask):
        print(f"⚠️  Found {np.sum(zero_mask)} zero feature vectors, setting to small values")
        norms[zero_mask] = 1e-8
    
    clip_features_norm = clip_features / norms
    
    # Compute cosine similarities
    similarities = np.dot(clip_features_norm, text_features.T).flatten()
    
    print(f"📈 Similarity stats - Min: {similarities.min():.3f}, Max: {similarities.max():.3f}, Mean: {similarities.mean():.3f}")
    
    # Filter by similarity threshold
    above_threshold = similarities >= similarity_threshold
    valid_indices = np.where(above_threshold)[0]
    
    if len(valid_indices) == 0:
        print(f"⚠️  No points above similarity threshold {similarity_threshold}")
        print(f"   Try lowering threshold or using different query")
        return np.array([]), np.array([]).reshape(0, 3), np.array([]).reshape(0, 3), similarities
    
    # Get top-k most similar points among those above threshold
    valid_similarities = similarities[valid_indices]
    if len(valid_indices) > top_k:
        # Sort by similarity and take top-k
        sorted_idx = np.argsort(valid_similarities)[::-1]
        top_valid_indices = valid_indices[sorted_idx[:top_k]]
    else:
        top_valid_indices = valid_indices
    
    highlight_indices = top_valid_indices
    highlight_points = points[highlight_indices]
    
    # Create highlight colors with intensity based on similarity
    highlight_similarities = similarities[highlight_indices]
    num_highlights = len(highlight_indices)
    
    # Create gradient colors based on similarity strength
    highlight_colors = np.zeros((num_highlights, 3))
    for i, sim in enumerate(highlight_similarities):
        # Scale similarity to color intensity (0.5 to 1.0 for visibility)
        intensity = 0.5 + 0.5 * (sim - similarity_threshold) / (similarities.max() - similarity_threshold)
        highlight_colors[i] = [c * intensity for c in highlight_color]
    
    print(f"✨ Highlighted {len(highlight_indices)} points with similarity > {similarity_threshold}")
    print(f"🎯 Top similarity: {highlight_similarities.max():.3f}")
    
    return highlight_indices, highlight_points, highlight_colors, similarities

def visualize_featurized_pointcloud(
    files_info,
    file_key='combined',
    mode="serve",
    remote_host=None, 
    remote_port=9878,
    show_features=True,
    feature_type='both',
    pca_method='hsv',
    point_size=0.01,
    create_mesh=False,
    mesh_method='poisson',
    use_voxels=False,
    voxel_size=0.05,
    enable_highlights=False,
    highlight_mode='random_points',
    animate_highlights_flag=False,
    animation_duration=30.0,
    text_query=None,
    text_similarity_top_k=100,
    text_similarity_threshold=0.3,
    clip_model_version="ViT-B/32"
):
    """Visualize featurized pointcloud data with enhanced interpolation, highlighting, and text-based similarity search options."""
    
    print(f"=== Starting Enhanced Featurized Pointcloud Visualization ===")
    print(f"Mode: {mode}")
    print(f"Remote host: {remote_host}")
    print(f"Remote port: {remote_port}")
    print(f"Show features: {show_features}")
    print(f"Feature type: {feature_type}")
    print(f"Point size: {point_size}")
    print(f"Create mesh: {create_mesh}")
    print(f"Mesh method: {mesh_method}")
    print(f"Use voxels: {use_voxels}")
    print(f"Enable highlights: {enable_highlights}")
    print(f"Highlight mode: {highlight_mode}")
    print(f"Animate highlights: {animate_highlights_flag}")
    if text_query:
        print(f"Text query: '{text_query}'")
        print(f"Text similarity top-k: {text_similarity_top_k}")
        print(f"Text similarity threshold: {text_similarity_threshold}")
        print(f"CLIP model: {clip_model_version}")
    
    if file_key not in files_info:
        print(f"Error: File key '{file_key}' not found. Available: {list(files_info.keys())}")
        return
    
    # Initialize CLIP encoder if text query is provided
    clip_encoder = None
    if text_query and HAS_CLIP_ENCODER:
        # Load pointcloud first to determine correct CLIP model
        try:
            print("🔍 Checking point cloud feature dimensions...")
            temp_points, temp_rgb, temp_features_info = load_featurized_pointcloud(files_info[file_key])
            
            # Auto-detect CLIP model version based on feature dimensions
            if 'clip' in temp_features_info:
                feature_dim = temp_features_info['clip'].shape[1]
                if feature_dim == 512:
                    detected_clip_version = "ViT-B/32"
                elif feature_dim == 768:
                    detected_clip_version = "ViT-L/14"
                else:
                    print(f"⚠️  Unknown CLIP feature dimension: {feature_dim}, defaulting to {clip_model_version}")
                    detected_clip_version = clip_model_version
                
                if detected_clip_version != clip_model_version:
                    print(f"🔄 Auto-detected CLIP model: {detected_clip_version} (feature dim: {feature_dim})")
                    print(f"   Overriding specified model: {clip_model_version}")
                    clip_model_version = detected_clip_version
                else:
                    print(f"✓ CLIP model {clip_model_version} matches feature dimension: {feature_dim}")
            
        except Exception as e:
            print(f"⚠️  Could not auto-detect CLIP model: {e}")
        
        try:
            print(f"🤖 Initializing CLIP encoder ({clip_model_version})...")
            clip_encoder = ClipEncoder(version=clip_model_version)
            print(f"✓ CLIP encoder ready on device: {clip_encoder.device}")
        except Exception as e:
            print(f"❌ Failed to initialize CLIP encoder: {e}")
            clip_encoder = None
    elif text_query and not HAS_CLIP_ENCODER:
        print("⚠️  Text query provided but CLIP encoder not available")
    
    rr.init("Enhanced_Featurized_Pointcloud", spawn=False)
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
            # Use serve_grpc for local mode too, with available port
            print(f"🖥️  Local mode: serving gRPC on port {remote_port}")
            rr.serve_grpc(grpc_port=remote_port)
            print(f"Connect with: rerun --connect rerun+http://127.0.0.1:{remote_port}/proxy")
    elif mode == "web":
        print("🌐 Using WEB mode - serving on localhost:9090")
        rr.serve_web(open_browser=True, web_port=9090)
    else:
        print(f"❌ Unknown mode: {mode}")
        
    print("✓ Viewer setup complete")

    # Set up coordinate frame and timeline
    print("🌍 Setting up world coordinate frame...")
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Y_UP, static=True)
    rr.set_time_seconds("timeline", 0.0)

    # Load the featurized pointcloud
    print("☁️  Loading featurized pointcloud...")
    points, rgb, features_info = load_featurized_pointcloud(files_info[file_key])

    # Apply voxel grid if requested
    if use_voxels:
        points, rgb = create_voxel_grid(points, rgb, voxel_size)

    # Log the original RGB pointcloud with larger points
    print(f"✓ Logging {len(points)} points with original RGB colors (size: {point_size})")
    rr.log("world/pointcloud_rgb", 
           rr.Points3D(points, colors=rgb, radii=point_size), 
           static=True)

    # Create mesh if requested
    vertices, faces, vertex_colors = None, None, None
    if create_mesh and HAS_OPEN3D:
        print("🔺 Creating surface mesh...")
        vertices, faces, vertex_colors = estimate_normals_and_mesh(
            points, rgb, method=mesh_method
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
    if show_features and features_info:
        print("🎨 Generating feature-based visualizations...")
        
        # Visualize CLIP features if available and requested
        if 'clip' in features_info and feature_type in ['clip', 'both']:
            print("🔍 Visualizing CLIP features...")
            clip_colors = features_to_colors_pca(features_info['clip'], method=pca_method)
            
            # Voxelize feature colors too if using voxels
            if use_voxels:
                _, clip_colors = create_voxel_grid(points, clip_colors, voxel_size)
            
            # Points
            rr.log("world/pointcloud_clip_features", 
                   rr.Points3D(points, colors=clip_colors, radii=point_size), 
                   static=True)
            
            # Mesh with CLIP features
            if create_mesh and HAS_OPEN3D and vertices is not None:
                clip_vertices, clip_faces, clip_vertex_colors = estimate_normals_and_mesh(
                    points, clip_colors, method=mesh_method
                )
                if clip_vertex_colors is not None:
                    rr.log("world/mesh_clip_features", 
                           rr.Mesh3D(vertex_positions=clip_vertices, 
                                   triangle_indices=clip_faces,
                                   vertex_colors=clip_vertex_colors), 
                           static=True)
            
            print(f"✓ CLIP feature visualization complete ({features_info['clip'].shape[1]}D → RGB)")
        
        # Visualize DINO features if available and requested  
        if 'dino' in features_info and feature_type in ['dino', 'both']:
            print("🦕 Visualizing DINO features...")
            dino_colors = features_to_colors_pca(features_info['dino'], method=pca_method)
            
            # Voxelize feature colors too if using voxels
            if use_voxels:
                _, dino_colors = create_voxel_grid(points, dino_colors, voxel_size)
            
            # Points
            rr.log("world/pointcloud_dino_features", 
                   rr.Points3D(points, colors=dino_colors, radii=point_size), 
                   static=True)
            
            # Mesh with DINO features
            if create_mesh and HAS_OPEN3D and vertices is not None:
                dino_vertices, dino_faces, dino_vertex_colors = estimate_normals_and_mesh(
                    points, dino_colors, method=mesh_method
                )
                if dino_vertex_colors is not None:
                    rr.log("world/mesh_dino_features", 
                           rr.Mesh3D(vertex_positions=dino_vertices, 
                                   triangle_indices=dino_faces,
                                   vertex_colors=dino_vertex_colors), 
                           static=True)
            
            print(f"✓ DINO feature visualization complete ({features_info['dino'].shape[1]}D → RGB)")

    # Add highlighting features
    if enable_highlights:
        print("🎯 Adding highlighting features...")
        
        # Text-based similarity highlighting
        if text_query and clip_encoder and 'clip' in features_info:
            print("🔍 Creating text-based similarity highlights...")
            highlight_indices, highlight_points, highlight_colors, similarities = create_text_similarity_highlights(
                points, features_info['clip'], text_query,
                clip_encoder=clip_encoder,
                top_k=text_similarity_top_k,
                similarity_threshold=text_similarity_threshold,
                highlight_color=[1.0, 0.8, 0.0]  # Golden yellow for text similarity
            )
            
            if len(highlight_points) > 0:
                # Log text similarity highlights with larger size
                rr.log("world/text_similarity_highlights", 
                       rr.Points3D(highlight_points, 
                                 colors=highlight_colors, 
                                 radii=0.03))
                
                # Log similarity statistics
                rr.log("text_search/query", rr.TextLog(f"Query: '{text_query}'"))
                rr.log("text_search/num_matches", rr.Scalar(len(highlight_points)))
                rr.log("text_search/top_similarity", rr.Scalar(float(similarities.max())))
                rr.log("text_search/mean_similarity", rr.Scalar(float(similarities.mean())))
                
                print(f"✨ Text similarity highlights: {len(highlight_points)} points found")
            else:
                print("⚠️  No points found matching the text query")
        
        elif highlight_mode == 'voxel_highlights':
            # Voxel-based highlighting
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
        
        else:
            # Static highlighting
            highlight_indices, highlight_points, highlight_colors = create_random_highlights(
                points, rgb, highlight_mode=highlight_mode, num_highlights=200
            )
            
            rr.log("world/static_highlights", 
                   rr.Points3D(highlight_points, colors=highlight_colors, radii=0.025), 
                   static=True)
        
        # Animated highlighting
        if animate_highlights_flag:
            animation_thread = animate_highlights(
                points, rgb, "world/pointcloud_rgb", 
                duration=animation_duration,
                highlight_interval=0.5,
                num_highlights=100,
                highlight_modes=['random_points', 'spatial_clusters', 'tight_clusters', 'feature_based']
            )

    # Log comprehensive statistics with better structure
    print("📊 Logging pointcloud statistics...")
    
    # Basic point cloud info
    bbox_min = points.min(axis=0)
    bbox_max = points.max(axis=0)
    bbox_size = bbox_max - bbox_min
    
    # Log individual scalar values properly
    rr.log("stats/num_points", rr.Scalar(len(points)), static=True)
    rr.log("stats/bbox_size_x", rr.Scalar(float(bbox_size[0])), static=True)
    rr.log("stats/bbox_size_y", rr.Scalar(float(bbox_size[1])), static=True)
    rr.log("stats/bbox_size_z", rr.Scalar(float(bbox_size[2])), static=True)
    rr.log("stats/bbox_volume", rr.Scalar(float(np.prod(bbox_size))), static=True)
    
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
            rr.log(f"stats/features_{feat_name}_dim", rr.Scalar(feat_dim), static=True)
            rr.log(f"stats/features_{feat_name}_mean_norm", rr.Scalar(float(feat_mean_norm)), static=True)
            rr.log(f"stats/features_{feat_name}_std_norm", rr.Scalar(float(feat_std_norm)), static=True)
            
            summary_text += f"   • {feat_name.upper()}: {feat_dim}D features, norm μ={feat_mean_norm:.3f} σ={feat_std_norm:.3f}\n"
    
    summary_text += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    # Log the comprehensive summary as a text log
    rr.log("stats/summary", rr.TextLog(summary_text, level=rr.TextLogLevel.INFO), static=True)
    
    # Also print to console for immediate reference
    print(summary_text)

    print("🎉 Enhanced featurized pointcloud visualization complete!")
    print("\n📋 Available views in Rerun:")
    print("  - world/pointcloud_rgb: Original RGB colors with enhanced points")
    if create_mesh:
        print("  - world/mesh_rgb: Surface mesh with RGB colors")
    if show_features and features_info:
        if 'clip' in features_info and feature_type in ['clip', 'both']:
            print("  - world/pointcloud_clip_features: CLIP features as colors")
            if create_mesh:
                print("  - world/mesh_clip_features: CLIP features mesh")
        if 'dino' in features_info and feature_type in ['dino', 'both']:
            print("  - world/pointcloud_dino_features: DINO features as colors")
            if create_mesh:
                print("  - world/mesh_dino_features: DINO features mesh")
    if enable_highlights:
        if text_query and clip_encoder:
            print("  - world/text_similarity_highlights: Text-based semantic similarity highlights")
            print("  - text_search/*: Text search statistics and info")
        elif highlight_mode == 'voxel_highlights':
            print("  - world/voxel_highlights: Highlighted voxel points")
            print("  - world/voxel_centers: Voxel center markers")
        else:
            print("  - world/static_highlights: Static random highlights")
        if animate_highlights_flag:
            print("  - world/pointcloud_rgb/highlights: Dynamic animated highlights")
    print("  - stats/summary: Comprehensive pointcloud summary with metrics and feature info")
    print("  - stats/*: Individual metric values (num_points, bbox dimensions, feature stats)")
    
    if remote_host:
        print(f"🌐 Streaming to {remote_host}:{remote_port}. Open the Rerun viewer and connect.")
    else:
        print("🖥️  Check the Rerun viewer window that should have opened.")

    if mode == "serve":
        if remote_host:
            print(f"\nVisualization server is running remotely on {remote_host}:{remote_port}.")
        else:
            print(f"\nVisualization server is running locally on port {remote_port}.")
            print(f"Connect with: rerun --connect rerun+http://127.0.0.1:{remote_port}/proxy")
        
        if text_query:
            print(f"\n🔍 Text-based semantic search for: '{text_query}'")
            print("💡 Golden/yellow highlights show points most similar to your query!")
        
        if animate_highlights_flag:
            print(f"\n🎬 Dynamic highlighting will run for {animation_duration} seconds.")
            print("💡 Tip: Use the timeline slider in Rerun to scrub through the animation!")
        
        print("Press Enter to exit the server.")
        input()

def main():
    parser = argparse.ArgumentParser(description="Visualize featurized pointclouds from MASt3R preprocessing with surface reconstruction and dynamic highlighting")
    parser.add_argument("pointcloud_dir", 
                       help="Directory containing featurized pointcloud .pt files")
    parser.add_argument("--file-type",
                       default="combined",
                       help="Which .pt file to visualize (use discovered file keys like 'clip', 'dino', 'combined', or actual filenames without extension)")
    parser.add_argument("--mode",
                       choices=["serve", "save", "web"],
                       default="serve",
                       help="Set the visualization mode: 'serve' to stream, 'save' to file, 'web' to serve on localhost:9090")
    parser.add_argument("--remote-host",
                       help="Remote host IP for Rerun streaming")
    parser.add_argument("--remote-port", type=int, default=9878)
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
    
    # Enhancement options
    parser.add_argument("--point-size", type=float, default=0.01,
                       help="Size of points for better visibility (default: 0.01)")
    parser.add_argument("--create-mesh", action="store_true",
                       help="Create surface mesh from pointcloud for better visualization")
    parser.add_argument("--mesh-method",
                       choices=["poisson", "ball_pivoting", "alpha_shape"],
                       default="poisson",
                       help="Surface reconstruction method")
    parser.add_argument("--use-voxels", action="store_true",
                       help="Use voxel grid for smoother representation")
    parser.add_argument("--voxel-size", type=float, default=0.05,
                       help="Voxel size for grid downsampling")
    
    # New highlighting options
    parser.add_argument("--enable-highlights", action="store_true",
                       help="Enable random highlighting of points/voxels")
    parser.add_argument("--highlight-mode",
                       choices=["random_points", "spatial_clusters", "tight_clusters", "feature_based", "voxel_highlights"],
                       default="random_points",
                       help="Method for selecting points to highlight")
    parser.add_argument("--animate-highlights", action="store_true",
                       help="Create animated highlighting effects over time")
    parser.add_argument("--animation-duration", type=float, default=30.0,
                       help="Duration of highlight animation in seconds")
    
    # Text-based similarity options
    parser.add_argument("--text-query",
                       help="Text query for text-based similarity highlighting")
    parser.add_argument("--text-similarity-top-k", type=int, default=100,
                       help="Number of top similar points to highlight")
    parser.add_argument("--text-similarity-threshold", type=float, default=0.3,
                       help="Minimum cosine similarity threshold for text-based similarity")
    parser.add_argument("--clip-model-version",
                       choices=["ViT-B/32", "ViT-L/14"],
                       default="ViT-B/32",
                       help="CLIP model version for text-based similarity")
    
    args = parser.parse_args()

    # Auto-discover featurized pointcloud files
    try:
        files_info = discover_featurized_files(args.pointcloud_dir)
    except Exception as e:
        print(f"Error discovering files: {e}")
        return

    # Validate file type selection
    if args.file_type not in files_info:
        print(f"Error: File type '{args.file_type}' not found.")
        print(f"Available file types: {list(files_info.keys())}")
        print(f"Use one of these keys with --file-type")
        return

    visualize_featurized_pointcloud(
        files_info,
        file_key=args.file_type,
        mode=args.mode,
        remote_host=args.remote_host, 
        remote_port=args.remote_port,
        show_features=not args.no_features,
        feature_type=args.feature_type,
        pca_method=args.pca_method,
        point_size=args.point_size,
        create_mesh=args.create_mesh,
        mesh_method=args.mesh_method,
        use_voxels=args.use_voxels,
        voxel_size=args.voxel_size,
        enable_highlights=args.enable_highlights,
        highlight_mode=args.highlight_mode,
        animate_highlights_flag=args.animate_highlights,
        animation_duration=args.animation_duration,
        text_query=args.text_query,
        text_similarity_top_k=args.text_similarity_top_k,
        text_similarity_threshold=args.text_similarity_threshold,
        clip_model_version=args.clip_model_version
    )

if __name__ == "__main__":
    main() 