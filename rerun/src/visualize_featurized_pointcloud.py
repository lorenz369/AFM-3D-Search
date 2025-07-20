#!/usr/bin/env python3
"""
Featurized Pointcloud visualization script using Rerun SDK
Visualizes the featurized point clouds from MASt3R preprocessing pipeline
with support for feature visualization through PCA color mapping, surface reconstruction,
and dynamic random highlighting
"""

import numpy as np
import torch
import os
import glob
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import open3d as o3d
from matplotlib.colors import hsv_to_rgb

__all__ = [
    "discover_featurized_files",
    "load_featurized_pointcloud",
    "features_to_colors_pca",
    "estimate_normals_and_mesh",
    "create_text_similarity_highlights",
    "visualize_original_pointcloud",
] 

def discover_featurized_files(base_dir):
    """Auto-discover featurized pointcloud files (.pt) or fallback to .npy/.ply in the directory."""
    import os
    import glob
    base_dir = os.path.abspath(base_dir)
    
    # Find all .pt files
    pt_files = glob.glob(os.path.join(base_dir, "*.pt"))
    if pt_files:
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
    
    # If no .pt files, look for .npy and .ply files
    npy_clip = os.path.join(base_dir, "clip_features.npy")
    npy_dino = os.path.join(base_dir, "dino_features.npy")
    ply_file = os.path.join(base_dir, "point_cloud.ply")
    files_info = {}
    found_any = False
    if os.path.isfile(npy_clip):
        files_info['clip'] = npy_clip
        found_any = True
    if os.path.isfile(npy_dino):
        files_info['dino'] = npy_dino
        found_any = True
    if os.path.isfile(ply_file):
        files_info['ply'] = ply_file
        found_any = True
    if found_any:
        print(f"Discovered featurized numpy/ply files in {base_dir}:")
        for key, path in files_info.items():
            file_size = os.path.getsize(path) / (1024**3)  # GB
            print(f"  {key}: {os.path.basename(path)} ({file_size:.1f} GB)")
        return files_info
    
    raise FileNotFoundError(f"No .pt or expected .npy/.ply files found in {base_dir}")

def load_featurized_pointcloud(pt_path):
    """Load featurized pointcloud from .pt file, or from numpy/ply files if given a dict."""
    import numpy as np
    import open3d as o3d
    import os
    import torch

    if isinstance(pt_path, dict):
        # Numpy/PLY variant: expects keys 'clip', 'dino', 'ply'
        files_info = pt_path
        print(f"Loading featurized pointcloud from numpy/ply files: {files_info}")
        # Load points and rgb from ply
        ply_path = files_info.get('ply')
        if ply_path is None or not os.path.isfile(ply_path):
            raise FileNotFoundError(f"PLY file not found in files_info: {ply_path}")
        pcd = o3d.io.read_point_cloud(ply_path)
        points = np.asarray(pcd.points)
        rgb = np.asarray(pcd.colors) if pcd.has_colors() else np.zeros_like(points)
        if rgb.max() > 1.0:
            rgb = rgb / 255.0
        features_info = {}
        # Load features
        if 'clip' in files_info and os.path.isfile(files_info['clip']):
            features_clip = np.load(files_info['clip'])
            features_info['clip'] = features_clip
            print(f"  CLIP features: {features_clip.shape}")
        else:
            print("  CLIP features: None")
        if 'dino' in files_info and os.path.isfile(files_info['dino']):
            features_dino = np.load(files_info['dino'])
            features_info['dino'] = features_dino
            print(f"  DINO features: {features_dino.shape}")
        else:
            print("  DINO features: None")
        print(f"Loaded pointcloud: {points.shape[0]} points, RGB shape: {rgb.shape}")
        return points, rgb, features_info
    elif isinstance(pt_path, str) and pt_path.endswith('.pt'):
        # Original .pt logic
        print(f"Loading featurized pointcloud from {pt_path}...")
        try:
            data = torch.load(pt_path, map_location='cpu')
            points = data['points'].numpy() if isinstance(data['points'], torch.Tensor) else data['points']
            rgb = data['rgb'].numpy() if isinstance(data['rgb'], torch.Tensor) else data['rgb']
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
    elif isinstance(pt_path, str) and pt_path.endswith('.npy'):
        # Fallback: load a single feature array (not recommended)
        print(f"Loading features from {pt_path} (no point cloud)")
        features = np.load(pt_path)
        points = np.zeros((features.shape[0], 3))
        rgb = np.zeros_like(points)
        features_info = {'clip': features}
        return points, rgb, features_info
    else:
        raise ValueError(f"Unsupported input to load_featurized_pointcloud: {pt_path}")

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
    
    if clip_encoder is None:
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

def load_original_pointcloud(file_key, original_pointcloud):
    """
    Load the original pointcloud (PLY) corresponding to a featurized pointcloud.
    Args:
        file_key: str, the key/scene id (e.g., '42447230')
        original_pointcloud: str, path to the Training directory (e.g., '/path/to/Training')
    Returns:
        points: np.ndarray of shape (N, 3)
        colors: np.ndarray of shape (N, 3) or None
    """
    # Determine the PLY file path
    ply_path = os.path.join(
        os.path.abspath(original_pointcloud),
        file_key,
        f"{file_key}_3dod_mesh.ply"
    )
    if not os.path.exists(ply_path):
        raise FileNotFoundError(f"Original PLY file not found: {ply_path}")

    print(f"Loading original pointcloud from {ply_path} ...")
    pcd = o3d.io.read_point_cloud(ply_path)
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors) if pcd.has_colors() else None
    if colors is not None and colors.max() > 1.0:
        colors = colors / 255.0

    print(f"Loaded {points.shape[0]} points from original PLY.")
    return points, colors