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
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb

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

def estimate_normals_and_mesh(points, colors=None, method='poisson', depth=9, density_threshold=0.1):
    """
    Create surface mesh from pointcloud using various reconstruction methods.
    
    Args:
        points: numpy array [N, 3]
        colors: numpy array [N, 3] optional
        method: 'poisson', 'ball_pivoting', or 'delaunay'
        depth: depth for Poisson reconstruction
        density_threshold: threshold for removing low-density vertices
    
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
        print("Running Alpha Shape reconstruction...")
        alpha = 0.03  # You may need to tune this
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)
    
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
    animation_duration=30.0
):
    """Visualize featurized pointcloud data with enhanced interpolation and highlighting options."""
    
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
    
    if file_key not in files_info:
        print(f"Error: File key '{file_key}' not found. Available: {list(files_info.keys())}")
        return
    
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
        
        if highlight_mode == 'voxel_highlights':
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

    # Log some basic statistics
    print("📊 Logging pointcloud statistics...")
    rr.log("stats/num_points", rr.Scalar(len(points)), static=True)
    rr.log("stats/bbox_min", rr.Scalar(points.min(axis=0)), static=True)
    rr.log("stats/bbox_max", rr.Scalar(points.max(axis=0)), static=True)
    
    if features_info:
        for feat_name, features in features_info.items():
            rr.log(f"stats/features_{feat_name}_dim", rr.Scalar(features.shape[1]), static=True)
            rr.log(f"stats/features_{feat_name}_mean_norm", rr.Scalar(np.linalg.norm(features, axis=1).mean()), static=True)

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
        if highlight_mode == 'voxel_highlights':
            print("  - world/voxel_highlights: Highlighted voxel points")
            print("  - world/voxel_centers: Voxel center markers")
        else:
            print("  - world/static_highlights: Static random highlights")
        if animate_highlights_flag:
            print("  - world/pointcloud_rgb/highlights: Dynamic animated highlights")
    print("  - stats/*: Various statistics")
    
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
                       choices=["clip", "dino", "combined"],
                       default="combined",
                       help="Which .pt file to visualize")
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
        pca_method=args.pca_method,
        point_size=args.point_size,
        create_mesh=args.create_mesh,
        mesh_method=args.mesh_method,
        use_voxels=args.use_voxels,
        voxel_size=args.voxel_size,
        enable_highlights=args.enable_highlights,
        highlight_mode=args.highlight_mode,
        animate_highlights_flag=args.animate_highlights,
        animation_duration=args.animation_duration
    )

if __name__ == "__main__":
    main() 