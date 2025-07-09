#!/usr/bin/env python3
"""
Interactive Text Search for Featurized Pointclouds using Rerun SDK

This script provides real-time interactive text search capabilities:
1. Terminal input: Type queries directly in the terminal
2. File watching: Edit query.txt file and see results update
3. Socket interface: Send queries via network socket

Usage:
  python visualize_interactive_text_search.py data/ARKitScenes_fpt --file-type 42447230

Controls:
  - Type in terminal for immediate search
  - Edit 'query.txt' file for file-based search
  - Press 'q' + Enter to quit
  - Press 'clear' + Enter to clear highlights
"""

import rerun as rr
import numpy as np
import torch
import argparse
import os
import sys
import time
import threading
import queue
import select
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import socket
import json

# Import visualization functions
from visualize_featurized_pointcloud import (
    discover_featurized_files,
    load_featurized_pointcloud,
    features_to_colors_pca,
    create_text_similarity_highlights,
    estimate_normals_and_mesh,  # Add mesh creation
    HAS_CLIP_ENCODER,
    HAS_OPEN3D  # Add open3d availability check
)

def detect_similarity_outliers(similarities, method='adaptive', min_threshold=0.1, 
                              percentile_threshold=95, iqr_multiplier=1.5, 
                              z_score_threshold=2.0, min_points=5):
    """
    Detect outlier high-similarity points using various statistical methods.
    
    Args:
        similarities: numpy array of similarity scores
        method: 'iqr', 'percentile', 'z_score', 'adaptive', or 'combined'
        min_threshold: minimum similarity to consider (filters noise)
        percentile_threshold: percentile threshold for percentile method
        iqr_multiplier: multiplier for IQR method
        z_score_threshold: threshold for z-score method
        min_points: minimum number of points to return
    
    Returns:
        outlier_indices: indices of outlier points
        threshold_used: the actual threshold that was applied
        method_used: the method that was actually used
        stats: dictionary with statistical information
    """
    
    # Filter out very low similarities first
    valid_mask = similarities >= min_threshold
    valid_similarities = similarities[valid_mask]
    valid_indices = np.where(valid_mask)[0]
    
    if len(valid_similarities) == 0:
        return np.array([]), min_threshold, 'none', {}
    
    # Calculate statistical measures
    mean_sim = np.mean(valid_similarities)
    median_sim = np.median(valid_similarities)
    std_sim = np.std(valid_similarities)
    q25, q75 = np.percentile(valid_similarities, [25, 75])
    iqr = q75 - q25
    
    stats = {
        'mean': mean_sim,
        'median': median_sim,
        'std': std_sim,
        'q25': q25,
        'q75': q75,
        'iqr': iqr,
        'min': valid_similarities.min(),
        'max': valid_similarities.max(),
        'count': len(valid_similarities)
    }
    
    if method == 'iqr':
        # IQR-based outlier detection
        threshold = q75 + iqr_multiplier * iqr
        outlier_mask = valid_similarities >= threshold
        method_used = 'iqr'
        
    elif method == 'percentile':
        # Percentile-based detection
        threshold = np.percentile(valid_similarities, percentile_threshold)
        outlier_mask = valid_similarities >= threshold
        method_used = 'percentile'
        
    elif method == 'z_score':
        # Z-score based detection
        z_scores = (valid_similarities - mean_sim) / (std_sim + 1e-8)
        threshold = mean_sim + z_score_threshold * std_sim
        outlier_mask = z_scores >= z_score_threshold
        method_used = 'z_score'
        
    elif method == 'adaptive':
        # Adaptive method: choose based on data characteristics
        if iqr > 0.05:  # If there's good spread, use IQR
            threshold = q75 + iqr_multiplier * iqr
            outlier_mask = valid_similarities >= threshold
            method_used = 'iqr'
        elif std_sim > 0.02:  # If there's decent variation, use z-score
            z_scores = (valid_similarities - mean_sim) / (std_sim + 1e-8)
            threshold = mean_sim + z_score_threshold * std_sim
            outlier_mask = z_scores >= z_score_threshold
            method_used = 'z_score'
        else:  # Fall back to percentile
            threshold = np.percentile(valid_similarities, percentile_threshold)
            outlier_mask = valid_similarities >= threshold
            method_used = 'percentile'
            
    elif method == 'combined':
        # Combined approach: use multiple methods and take intersection
        # IQR outliers
        iqr_threshold = q75 + iqr_multiplier * iqr
        iqr_outliers = valid_similarities >= iqr_threshold
        
        # Z-score outliers
        z_scores = (valid_similarities - mean_sim) / (std_sim + 1e-8)
        z_outliers = z_scores >= z_score_threshold
        
        # Percentile outliers
        perc_threshold = np.percentile(valid_similarities, percentile_threshold)
        perc_outliers = valid_similarities >= perc_threshold
        
        # Take union of methods (points that are outliers by ANY method)
        outlier_mask = iqr_outliers | z_outliers | perc_outliers
        threshold = min(iqr_threshold, mean_sim + z_score_threshold * std_sim, perc_threshold)
        method_used = 'combined'
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Get outlier indices
    outlier_indices_in_valid = np.where(outlier_mask)[0]
    outlier_indices = valid_indices[outlier_indices_in_valid]
    
    # Ensure minimum number of points if available
    if len(outlier_indices) < min_points and len(valid_indices) >= min_points:
        # Fall back to top-k approach
        top_k_indices = valid_indices[np.argsort(valid_similarities)[-min_points:]]
        outlier_indices = top_k_indices
        threshold = valid_similarities[np.argsort(valid_similarities)[-min_points]]
        method_used = f'{method_used}_topk_fallback'
    
    stats['threshold_used'] = threshold
    stats['num_outliers'] = len(outlier_indices)
    stats['outlier_percentage'] = (len(outlier_indices) / len(similarities)) * 100
    
    return outlier_indices, threshold, method_used, stats

def create_statistical_text_similarity_highlights(points, clip_features, text_query, 
                                                 clip_encoder=None, 
                                                 outlier_method='adaptive',
                                                 min_threshold=0.1,
                                                 percentile_threshold=95,
                                                 iqr_multiplier=1.5,
                                                 z_score_threshold=2.0,
                                                 min_points=5,
                                                 highlight_color=[1.0, 0.8, 0.0]):
    """
    Create highlights based on statistical outlier detection of CLIP similarities.
    
    This is an enhanced version that uses statistical methods to automatically 
    identify the high-similarity outlier group instead of fixed thresholds.
    """
    
    if not HAS_CLIP_ENCODER or clip_encoder is None:
        print("⚠️  CLIP encoder not available for text similarity highlighting")
        return np.array([]), np.array([]).reshape(0, 3), np.array([]).reshape(0, 3), np.array([]), {}
    
    if clip_features is None or len(clip_features) == 0:
        print("⚠️  No CLIP features available for text similarity highlighting")
        return np.array([]), np.array([]).reshape(0, 3), np.array([]).reshape(0, 3), np.array([]), {}
    
    print(f"🔍 Computing statistical text similarity for query: '{text_query}'")
    print(f"📊 CLIP features shape: {clip_features.shape}")
    print(f"📈 Outlier detection method: {outlier_method}")
    
    # Encode the text query (reuse logic from original function)
    try:
        text_features = clip_encoder.encode_text(text_query)
        text_features = text_features.cpu().numpy()
        text_features = text_features / np.linalg.norm(text_features, axis=1, keepdims=True)
        print(f"✓ Text encoded to {text_features.shape}")
        
    except Exception as e:
        print(f"❌ Error encoding text query: {e}")
        return np.array([]), np.array([]).reshape(0, 3), np.array([]).reshape(0, 3), np.array([]), {}
    
    # Check dimension compatibility
    clip_feature_dim = clip_features.shape[1]
    text_feature_dim = text_features.shape[1]
    
    if clip_feature_dim != text_feature_dim:
        print(f"❌ Dimension mismatch: CLIP features ({clip_feature_dim}D) vs Text features ({text_feature_dim}D)")
        return np.array([]), np.array([]).reshape(0, 3), np.array([]).reshape(0, 3), np.array([]), {}
    
    # Normalize CLIP features and compute similarities
    norms = np.linalg.norm(clip_features, axis=1, keepdims=True)
    zero_mask = norms[:, 0] == 0
    if np.any(zero_mask):
        print(f"⚠️  Found {np.sum(zero_mask)} zero feature vectors, setting to small values")
        norms[zero_mask] = 1e-8
    
    clip_features_norm = clip_features / norms
    similarities = np.dot(clip_features_norm, text_features.T).flatten()
    
    print(f"📈 Similarity stats - Min: {similarities.min():.3f}, Max: {similarities.max():.3f}, Mean: {similarities.mean():.3f}")
    
    # Apply statistical outlier detection
    outlier_indices, threshold_used, method_used, outlier_stats = detect_similarity_outliers(
        similarities, 
        method=outlier_method,
        min_threshold=min_threshold,
        percentile_threshold=percentile_threshold,
        iqr_multiplier=iqr_multiplier,
        z_score_threshold=z_score_threshold,
        min_points=min_points
    )
    
    if len(outlier_indices) == 0:
        print(f"⚠️  No statistical outliers found for '{text_query}'")
        print(f"   Max similarity: {similarities.max():.3f}, Method: {method_used}")
        return np.array([]), np.array([]).reshape(0, 3), np.array([]).reshape(0, 3), similarities, outlier_stats
    
    # Create highlights for outlier points
    highlight_points = points[outlier_indices]
    highlight_similarities = similarities[outlier_indices]
    
    # Create gradient colors based on similarity strength within outliers
    num_highlights = len(outlier_indices)
    highlight_colors = np.zeros((num_highlights, 3))
    
    # Normalize within outlier group for color intensity
    min_outlier_sim = highlight_similarities.min()
    max_outlier_sim = highlight_similarities.max()
    
    for i, sim in enumerate(highlight_similarities):
        if max_outlier_sim > min_outlier_sim:
            # Scale similarity to color intensity within outlier group
            intensity = 0.4 + 0.6 * (sim - min_outlier_sim) / (max_outlier_sim - min_outlier_sim)
        else:
            intensity = 1.0
        highlight_colors[i] = [c * intensity for c in highlight_color]
    
    print(f"✨ Statistical outlier detection results:")
    print(f"   Method used: {method_used}")
    print(f"   Threshold: {threshold_used:.3f}")
    print(f"   Outliers found: {len(outlier_indices)} ({outlier_stats['outlier_percentage']:.1f}% of total)")
    print(f"   Outlier similarities: {highlight_similarities.min():.3f} - {highlight_similarities.max():.3f}")
    
    return outlier_indices, highlight_points, highlight_colors, similarities, outlier_stats

def cluster_points_by_dino_features(points, dino_features, indices, n_clusters='auto', 
                                   spatial_weight=0.3, min_cluster_size=10):
    """
    Cluster points based on DINO features to identify coherent structures.
    
    Args:
        points: numpy array [N, 3] - 3D point coordinates
        dino_features: numpy array [N, D] - DINO features for all points
        indices: numpy array - indices of points to cluster (e.g., CLIP outliers)
        n_clusters: int or 'auto' - number of clusters
        spatial_weight: float - weight for spatial coordinates in clustering
        min_cluster_size: int - minimum points per cluster to keep
    
    Returns:
        filtered_indices: indices of points in the largest/best clusters
        cluster_info: dictionary with clustering information
    """
    
    if len(indices) < min_cluster_size:
        return indices, {'method': 'too_few_points', 'n_clusters': 0}
    
    # Extract features and coordinates for selected points
    selected_points = points[indices]
    selected_dino_features = dino_features[indices]
    
    # Normalize spatial coordinates to [0,1] range
    spatial_range = selected_points.max(axis=0) - selected_points.min(axis=0)
    spatial_range[spatial_range == 0] = 1  # Avoid division by zero
    normalized_spatial = (selected_points - selected_points.min(axis=0)) / spatial_range
    
    # Normalize DINO features
    normalized_dino = selected_dino_features / (np.linalg.norm(selected_dino_features, axis=1, keepdims=True) + 1e-8)
    
    # Combine spatial and feature information
    combined_features = np.concatenate([
        normalized_dino,
        normalized_spatial * spatial_weight
    ], axis=1)
    
    # Determine number of clusters
    if n_clusters == 'auto':
        # Use elbow method or estimate based on data size
        n_clusters = min(max(2, len(indices) // 20), 8)
    
    try:
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score
        
        # Try different cluster numbers and pick best
        best_clusters = 2
        best_score = -1
        
        for k in range(2, min(n_clusters + 1, len(indices) // min_cluster_size + 1)):
            if k >= len(indices):
                break
                
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(combined_features)
            
            if len(np.unique(cluster_labels)) > 1:
                score = silhouette_score(combined_features, cluster_labels)
                if score > best_score:
                    best_score = score
                    best_clusters = k
        
        # Final clustering with best number of clusters
        kmeans = KMeans(n_clusters=best_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(combined_features)
        
        # Find largest clusters that meet minimum size requirement
        unique_labels, counts = np.unique(cluster_labels, return_counts=True)
        valid_clusters = unique_labels[counts >= min_cluster_size]
        
        if len(valid_clusters) == 0:
            # If no clusters meet size requirement, return largest cluster
            largest_cluster = unique_labels[np.argmax(counts)]
            cluster_mask = cluster_labels == largest_cluster
        else:
            # Take all valid clusters (or top 2-3 largest)
            top_clusters = valid_clusters[np.argsort(counts[np.isin(unique_labels, valid_clusters)])[-3:]]
            cluster_mask = np.isin(cluster_labels, top_clusters)
        
        filtered_indices = indices[cluster_mask]
        
        cluster_info = {
            'method': 'kmeans',
            'n_clusters': best_clusters,
            'silhouette_score': best_score,
            'clusters_kept': len(np.unique(cluster_labels[cluster_mask])),
            'points_filtered': len(indices) - len(filtered_indices),
            'spatial_weight': spatial_weight
        }
        
        return filtered_indices, cluster_info
        
    except ImportError:
        print("⚠️  scikit-learn not available for clustering, using spatial fallback")
        # Fallback: spatial density-based filtering
        return spatial_density_filter(points, indices, min_cluster_size), {'method': 'spatial_fallback'}

def spatial_density_filter(points, indices, min_neighbors=5, radius_factor=0.02):
    """
    Fallback method: filter points based on spatial density.
    """
    if len(indices) < min_neighbors:
        return indices
    
    selected_points = points[indices]
    
    # Calculate bounding box to determine appropriate radius
    bbox_size = np.linalg.norm(selected_points.max(axis=0) - selected_points.min(axis=0))
    radius = bbox_size * radius_factor
    
    # Count neighbors for each point
    from scipy.spatial.distance import cdist
    distances = cdist(selected_points, selected_points)
    neighbor_counts = (distances < radius).sum(axis=1)
    
    # Keep points with enough neighbors
    dense_mask = neighbor_counts >= min_neighbors
    
    if dense_mask.sum() < min_neighbors:
        # If too few points remain, keep top half by neighbor count
        top_half = len(indices) // 2
        top_indices = np.argsort(neighbor_counts)[-top_half:]
        dense_mask = np.zeros(len(indices), dtype=bool)
        dense_mask[top_indices] = True
    
    return indices[dense_mask]

def create_hybrid_clip_dino_highlights(points, clip_features, dino_features, text_query, 
                                     clip_encoder=None, 
                                     outlier_method='adaptive',
                                     min_threshold=0.1,
                                     use_dino_filtering=True,
                                     dino_spatial_weight=0.3,
                                     min_cluster_size=10,
                                     max_clusters='auto',
                                     highlight_color=[1.0, 0.8, 0.0]):
    """
    Enhanced text similarity highlighting that combines CLIP semantic matching 
    with DINO structural filtering for more coherent results.
    
    Process:
    1. Use CLIP features to find semantically similar points (statistical outliers)
    2. Use DINO features to cluster and filter for structural coherence
    3. Return the most structurally consistent semantic matches
    """
    
    if not HAS_CLIP_ENCODER or clip_encoder is None:
        print("⚠️  CLIP encoder not available for text similarity highlighting")
        return np.array([]), np.array([]).reshape(0, 3), np.array([]).reshape(0, 3), np.array([]), {}
    
    if clip_features is None or len(clip_features) == 0:
        print("⚠️  No CLIP features available for text similarity highlighting")
        return np.array([]), np.array([]).reshape(0, 3), np.array([]).reshape(0, 3), np.array([]), {}
    
    print(f"🔍 Hybrid CLIP+DINO search for query: '{text_query}'")
    print(f"📊 CLIP features: {clip_features.shape}")
    if dino_features is not None and use_dino_filtering:
        print(f"🦕 DINO features: {dino_features.shape}")
        print(f"📈 Will use DINO for structural filtering")
    else:
        print("📈 DINO filtering disabled - using CLIP-only approach")
    
    # Step 1: Get CLIP semantic outliers (reuse existing function)
    clip_outlier_indices, threshold_used, method_used, outlier_stats = detect_similarity_outliers(
        np.dot(
            clip_features / (np.linalg.norm(clip_features, axis=1, keepdims=True) + 1e-8),
            clip_encoder.encode_text(text_query).cpu().numpy().T / np.linalg.norm(clip_encoder.encode_text(text_query).cpu().numpy())
        ).flatten(),
        method=outlier_method,
        min_threshold=min_threshold
    )
    
    if len(clip_outlier_indices) == 0:
        print(f"⚠️  No CLIP outliers found for '{text_query}'")
        return np.array([]), np.array([]).reshape(0, 3), np.array([]).reshape(0, 3), np.array([]), outlier_stats
    
    print(f"✓ CLIP found {len(clip_outlier_indices)} semantic outliers")
    
    # Step 2: Apply DINO structural filtering if available and enabled
    if dino_features is not None and use_dino_filtering and len(clip_outlier_indices) > min_cluster_size:
        print(f"🦕 Applying DINO structural filtering...")
        
        filtered_indices, cluster_info = cluster_points_by_dino_features(
            points, dino_features, clip_outlier_indices,
            n_clusters=max_clusters,
            spatial_weight=dino_spatial_weight,
            min_cluster_size=min_cluster_size
        )
        
        print(f"✓ DINO filtering: {len(clip_outlier_indices)} → {len(filtered_indices)} points")
        print(f"   Method: {cluster_info.get('method', 'unknown')}")
        if 'points_filtered' in cluster_info:
            print(f"   Filtered out: {cluster_info['points_filtered']} scattered points")
            print(f"   Kept clusters: {cluster_info.get('clusters_kept', 'unknown')}")
        
        final_indices = filtered_indices
        outlier_stats.update({
            'dino_filtering_enabled': True,
            'dino_cluster_info': cluster_info,
            'points_before_dino': len(clip_outlier_indices),
            'points_after_dino': len(filtered_indices)
        })
        
    else:
        final_indices = clip_outlier_indices
        outlier_stats.update({
            'dino_filtering_enabled': False,
            'reason': 'disabled' if not use_dino_filtering else 'not_available' if dino_features is None else 'too_few_points'
        })
    
    if len(final_indices) == 0:
        print(f"⚠️  No points remaining after filtering")
        return np.array([]), np.array([]).reshape(0, 3), np.array([]).reshape(0, 3), np.array([]), outlier_stats
    
    # Step 3: Create highlights for final filtered points
    highlight_points = points[final_indices]
    
    # Compute similarities for color mapping
    text_features = clip_encoder.encode_text(text_query).cpu().numpy()
    text_features = text_features / np.linalg.norm(text_features)
    clip_features_norm = clip_features / (np.linalg.norm(clip_features, axis=1, keepdims=True) + 1e-8)
    similarities = np.dot(clip_features_norm, text_features.T).flatten()
    
    highlight_similarities = similarities[final_indices]
    
    # Create gradient colors based on similarity strength
    num_highlights = len(final_indices)
    highlight_colors = np.zeros((num_highlights, 3))
    
    min_sim = highlight_similarities.min()
    max_sim = highlight_similarities.max()
    
    for i, sim in enumerate(highlight_similarities):
        if max_sim > min_sim:
            intensity = 0.4 + 0.6 * (sim - min_sim) / (max_sim - min_sim)
        else:
            intensity = 1.0
        highlight_colors[i] = [c * intensity for c in highlight_color]
    
    print(f"✨ Hybrid CLIP+DINO results:")
    print(f"   Final points: {len(final_indices)} ({len(final_indices)/len(points)*100:.1f}% of total)")
    print(f"   Similarity range: {highlight_similarities.min():.3f} - {highlight_similarities.max():.3f}")
    
    return final_indices, highlight_points, highlight_colors, similarities, outlier_stats

# Add the path to locate-3d for importing ClipEncoder
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'locate-3d'))
try:
    from preprocessing.image_features.clip_encoder import ClipEncoder
except ImportError:
    print("❌ ClipEncoder not available. This script requires CLIP functionality.")
    sys.exit(1)

class QueryFileHandler(FileSystemEventHandler):
    """Handler for watching query file changes."""
    
    def __init__(self, query_queue, query_file):
        self.query_queue = query_queue
        self.query_file = Path(query_file).name
        
    def on_modified(self, event):
        if event.is_directory:
            return
            
        if Path(event.src_path).name == self.query_file:
            try:
                with open(event.src_path, 'r') as f:
                    query = f.read().strip()
                if query:
                    self.query_queue.put(("file", query))
                    print(f"📁 File query: '{query}'")
            except Exception as e:
                print(f"❌ Error reading query file: {e}")

class InteractiveTextSearch:
    """Main class for interactive text search visualization with statistical outlier detection and DINO structural filtering."""
    
    def __init__(self, files_info, file_key, clip_model_version="ViT-B/32", create_mesh=True, 
                 outlier_method='adaptive', use_statistical_outliers=True, use_dino_filtering=True):
        self.files_info = files_info
        self.file_key = file_key
        self.clip_model_version = clip_model_version
        self.create_mesh = create_mesh
        self.outlier_method = outlier_method
        self.use_statistical_outliers = use_statistical_outliers
        self.use_dino_filtering = use_dino_filtering
        
        # Load data once
        print("🔄 Loading pointcloud data...")
        self.points, self.rgb, self.features_info = load_featurized_pointcloud(files_info[file_key])
        
        if 'clip' not in self.features_info:
            raise ValueError("❌ No CLIP features found in pointcloud. Text search requires CLIP features.")
        
        print(f"✅ Loaded {len(self.points)} points with CLIP features")
        
        # Check for DINO features
        self.has_dino_features = 'dino' in self.features_info and self.features_info['dino'] is not None
        if self.has_dino_features:
            print(f"🦕 DINO features available: {self.features_info['dino'].shape}")
            if use_dino_filtering:
                print("🔗 Hybrid CLIP+DINO mode enabled for structural filtering")
            else:
                print("📊 DINO features available but filtering disabled")
        else:
            print("⚠️  No DINO features found - using CLIP-only mode")
            self.use_dino_filtering = False
        
        if use_statistical_outliers:
            print(f"📈 Using statistical outlier detection method: {outlier_method}")
        else:
            print("🎯 Using traditional fixed threshold method")
        
        # Initialize CLIP encoder
        print("🤖 Initializing CLIP encoder...")
        self.clip_encoder = self._initialize_clip_encoder()
        
        # Setup queues and control
        self.query_queue = queue.Queue()
        self.running = True
        self.current_query = ""
        self.last_similarities = None
        self.last_outlier_stats = {}

        # Monotonically increasing counter used to time‐stamp every processed text
        # query. Each new query advances the global "timeline" so that dynamic
        # statistics such as similarity scores and outlier metrics are shown
        # as a proper 1-D time-series instead of collapsing at *t = 0*.
        #
        #   timeline = 0  → initial state  (point-cloud metadata)
        #   timeline = 1  → first user query
        #   timeline = 2  → second query, …
        #
        # This makes the built-in Time-Series View plots usable and avoids the
        # “vertical line at 1970-01-01" problem that happens when everything
        # is logged at the same instant with `static=True`.
        self.query_counter = 0
        
        # Query file path
        self.query_file = Path("query.txt")
        self._setup_query_file()
        
        # Pre-compute mesh for base pointcloud if requested
        self.base_mesh_vertices = None
        self.base_mesh_faces = None
        self.base_mesh_colors = None
        if self.create_mesh and HAS_OPEN3D:
            print("🔺 Pre-computing base mesh...")
            self.base_mesh_vertices, self.base_mesh_faces, self.base_mesh_colors = estimate_normals_and_mesh(
                self.points, self.rgb, method='ball_pivoting'
            )
    
    def _initialize_clip_encoder(self):
        """Initialize CLIP encoder with auto-detection."""
        feature_dim = self.features_info['clip'].shape[1]
        
        if feature_dim == 512:
            detected_version = "ViT-B/32"
        elif feature_dim == 768:
            detected_version = "ViT-L/14"
        else:
            detected_version = self.clip_model_version
            
        print(f"🔍 Auto-detected CLIP model: {detected_version} (feature dim: {feature_dim})")
        
        try:
            clip_encoder = ClipEncoder(version=detected_version)
            print(f"✅ CLIP encoder ready on device: {clip_encoder.device}")
            return clip_encoder
        except Exception as e:
            print(f"❌ Failed to initialize CLIP encoder: {e}")
            raise
    
    def _setup_query_file(self):
        """Setup the query file for file-based input."""
        if not self.query_file.exists():
            with open(self.query_file, 'w') as f:
                f.write("chair\n")
            print(f"📝 Created query file: {self.query_file}")
        else:
            print(f"📝 Using existing query file: {self.query_file}")
    
    def start_file_watcher(self):
        """Start file watcher for query.txt."""
        event_handler = QueryFileHandler(self.query_queue, self.query_file)
        observer = Observer()
        observer.schedule(event_handler, path=str(self.query_file.parent), recursive=False)
        observer.start()
        print(f"👀 Watching {self.query_file} for changes...")
        return observer
    
    def start_terminal_input(self):
        """Start terminal input thread."""
        def terminal_input_thread():
            print("\n" + "="*60)
            print("🎯 INTERACTIVE TEXT SEARCH READY")
            print("="*60)
            print("💡 How to search:")
            print(f"   • Type queries directly here and press Enter")
            print(f"   • Edit '{self.query_file}' file in any text editor")
            print("   • Type 'clear' to remove highlights")
            print("   • Type 'q' to quit")
            print("   • Type 'help' for more commands")
            print("="*60)
            
            while self.running:
                try:
                    # Use select for non-blocking input on Unix systems
                    if hasattr(select, 'select'):
                        ready, _, _ = select.select([sys.stdin], [], [], 0.1)
                        if ready:
                            query = sys.stdin.readline().strip()
                        else:
                            continue
                    else:
                        # Fallback for Windows
                        query = input("🔍 Enter search query: ").strip()
                    
                    if query:
                        if query.lower() == 'q':
                            self.running = False
                            break
                        elif query.lower() == 'clear':
                            self.query_queue.put(("terminal", ""))
                        elif query.lower() == 'help':
                            self._show_help()
                        elif query.lower().startswith('threshold='):
                            threshold = float(query.split('=')[1])
                            self.query_queue.put(("threshold", threshold))
                        elif query.lower().startswith('topk='):
                            top_k = int(query.split('=')[1])
                            self.query_queue.put(("topk", top_k))
                        elif query.lower().startswith('outlier_method='):
                            outlier_method = query.split('=')[1]
                            self.query_queue.put(("outlier_method", outlier_method))
                        elif query.lower().startswith('use_statistical_outliers='):
                            use_statistical_outliers = query.split('=')[1].lower() == 'true'
                            self.query_queue.put(("use_statistical_outliers", use_statistical_outliers))
                        elif query.lower().startswith('use_dino_filtering='):
                            use_dino_filtering = query.split('=')[1].lower() == 'true'
                            self.query_queue.put(("use_dino_filtering", use_dino_filtering))
                        else:
                            self.query_queue.put(("terminal", query))
                            print(f"🔍 Searching for: '{query}'")
                            
                except (EOFError, KeyboardInterrupt):
                    self.running = False
                    break
                except Exception as e:
                    print(f"❌ Input error: {e}")
        
        thread = threading.Thread(target=terminal_input_thread, daemon=True)
        thread.start()
        print("⌨️  Terminal input thread started")
        return thread
    
    def _show_help(self):
        """Show help information."""
        print("\n" + "="*70)
        print("📚 INTERACTIVE SEARCH HELP - WITH CLIP+DINO HYBRID FILTERING")
        print("="*70)
        print("🔍 Search Commands:")
        print("  • <text>                    - Search for text (e.g., 'chair', 'red sofa')")
        print("  • clear                     - Clear all highlights")
        print("  • q                         - Quit")
        print("")
        print("🔗 Hybrid CLIP+DINO Filtering:")
        if self.has_dino_features:
            print("  • use_dino_filtering=true   - Enable DINO structural filtering (default)")
            print("  • use_dino_filtering=false  - Use CLIP-only semantic search")
            print("    ↳ DINO helps identify coherent structures vs scattered points")
            print("    ↳ Shows both: CLIP outliers (blue) + DINO filtered (gold)")
        else:
            print("  • DINO features not available - using CLIP-only mode")
        print("")
        print("📈 Statistical Outlier Detection:")
        print("  • outlier_method=adaptive   - Auto-select best method based on data")
        print("  • outlier_method=iqr        - Use IQR (Interquartile Range) method")
        print("  • outlier_method=percentile - Use percentile-based detection")
        print("  • outlier_method=z_score    - Use Z-score method")
        print("  • outlier_method=combined   - Use combined methods")
        print("  • use_statistical_outliers=true/false - Toggle statistical mode")
        print("")
        print("🎯 Traditional Method Settings:")
        print("  • threshold=0.25            - Set fixed similarity threshold")
        print("  • topk=100                  - Set maximum number of results")
        print("")
        print("💡 Method Comparison:")
        print("  • Hybrid CLIP+DINO: Best results - semantic + structural coherence")
        print("    - CLIP finds semantic matches, DINO filters for structures")
        print("    - Reduces scattered points, focuses on object boundaries")
        print("  • Statistical CLIP-only: Good semantic matching with outlier detection")
        print("    - Adapts to object size automatically")
        print("  • Traditional: Fixed threshold, predictable but may need tuning")
        print("")
        print("🎮 Examples:")
        print("  • chair                     - Basic hybrid search")
        print("  • red wooden table          - Multi-word hybrid search")
        print("  • use_dino_filtering=false  - Switch to CLIP-only")
        print("  • outlier_method=iqr        - Switch outlier detection method")
        print("  • use_statistical_outliers=false - Switch to traditional")
        print("  • threshold=0.15            - Lower traditional threshold")
        print("="*70 + "\n")
    
    def process_text_query(self, query, top_k=200, threshold=0.2, outlier_method=None, use_statistical_outliers=None, use_dino_filtering=None):
        """Process a text query and return highlights using statistical or traditional methods."""
        
        # Use instance defaults if not specified
        if outlier_method is None:
            outlier_method = self.outlier_method
        if use_statistical_outliers is None:
            use_statistical_outliers = self.use_statistical_outliers
        if use_dino_filtering is None:
            use_dino_filtering = self.use_dino_filtering
            
        if not query or not query.strip():
            # Clear highlights
            rr.log("world/text_similarity_highlights", rr.Clear())
            rr.log("world/clip_semantic_outliers", rr.Clear())
            rr.log("world/highlighted_mesh", rr.Clear())
            
            # Clear search stats with better formatting
            rr.log("stats/search/current_query", rr.TextLog("No active search", level=rr.TextLogLevel.INFO))
            rr.log("stats/search/num_results", rr.Scalars(0))
            rr.log("stats/search/top_similarity", rr.Scalars(0.0))
            rr.log("stats/search/mean_similarity", rr.Scalars(0.0))
            rr.log("stats/search/threshold", rr.Scalars(threshold))
            rr.log("stats/search/detection_method", rr.TextLog("None"))
            return
        
        try:
            # ------------------------------------------------------------------
            # Advance time *once* per non-empty query so that all dynamic stats
            # share a meaningful x-axis in the Time-Series View.
            # ------------------------------------------------------------------
            if query and query.strip():
                self.query_counter += 1
                # Use the built-in "timeline" for simplicity – this is the same
                # timeline that was initialised to 0.0 in `run_interactive_session`.
                rr.set_time_seconds("timeline", float(self.query_counter))

            # Note: for an empty query ("clear") we *don’t* advance the counter –
            # this keeps the visualised stats aligned with the last executed
            # query.

            if use_statistical_outliers:
                # Choose method based on DINO availability and user preference
                if use_dino_filtering and self.has_dino_features:
                    # Use hybrid CLIP+DINO approach
                    highlight_indices, highlight_points, highlight_colors, similarities, outlier_stats = create_hybrid_clip_dino_highlights(
                        self.points, 
                        self.features_info['clip'], 
                        self.features_info['dino'],
                        query,
                        clip_encoder=self.clip_encoder,
                        outlier_method=outlier_method,
                        min_threshold=0.05,
                        use_dino_filtering=True,
                        dino_spatial_weight=0.3,
                        min_cluster_size=8,
                        highlight_color=[1.0, 0.8, 0.0]
                    )
                    method_used = f"hybrid_clip_dino_{outlier_stats.get('method_used', outlier_method)}"
                else:
                    # Use CLIP-only statistical outlier detection
                    highlight_indices, highlight_points, highlight_colors, similarities, outlier_stats = create_statistical_text_similarity_highlights(
                        self.points, 
                        self.features_info['clip'], 
                        query,
                        clip_encoder=self.clip_encoder,
                        outlier_method=outlier_method,
                        min_threshold=0.05,
                        highlight_color=[1.0, 0.8, 0.0]
                    )
                    method_used = outlier_stats.get('method_used', outlier_method) if outlier_stats else outlier_method
                
                self.last_similarities = similarities
                self.last_outlier_stats = outlier_stats
                
                # Determine threshold for display (from statistical method)
                display_threshold = outlier_stats.get('threshold_used', 0.0)
                
            else:
                # Use traditional fixed threshold method
                highlight_indices, highlight_points, highlight_colors, similarities = create_text_similarity_highlights(
                    self.points, 
                    self.features_info['clip'], 
                    query,
                    clip_encoder=self.clip_encoder,
                    top_k=top_k,
                    similarity_threshold=threshold,
                    highlight_color=[1.0, 0.8, 0.0]
                )
                
                self.last_similarities = similarities
                self.last_outlier_stats = {}
                display_threshold = threshold
                method_used = f"fixed_threshold_{top_k}"
            
            if len(highlight_points) > 0:
                # Update highlights in Rerun with larger points
                rr.log("world/text_similarity_highlights", 
                       rr.Points3D(highlight_points, colors=highlight_colors, radii=0.03))
                
                # If using hybrid CLIP+DINO, also show the original CLIP outliers for comparison
                if (use_statistical_outliers and use_dino_filtering and self.has_dino_features and 
                    self.last_outlier_stats.get('dino_filtering_enabled', False)):
                    
                    # Show original CLIP semantic outliers in a different color (blue-ish)
                    points_before_dino = self.last_outlier_stats.get('points_before_dino', 0)
                    if points_before_dino > 0:
                        # Get the original CLIP outlier indices from the similarity computation
                        # We need to recompute this since we only stored the final filtered results
                        text_features = self.clip_encoder.encode_text(query).cpu().numpy()
                        text_features = text_features / np.linalg.norm(text_features)
                        clip_features_norm = self.features_info['clip'] / (np.linalg.norm(self.features_info['clip'], axis=1, keepdims=True) + 1e-8)
                        similarities_temp = np.dot(clip_features_norm, text_features.T).flatten()
                        
                        clip_outlier_indices_temp, _, _, _ = detect_similarity_outliers(
                            similarities_temp, 
                            method=outlier_method,
                            min_threshold=0.05
                        )
                        
                        if len(clip_outlier_indices_temp) > 0:
                            clip_outlier_points = self.points[clip_outlier_indices_temp]
                            clip_outlier_similarities = similarities_temp[clip_outlier_indices_temp]
                            
                            # Create blue gradient colors for CLIP outliers
                            num_clip_outliers = len(clip_outlier_indices_temp)
                            clip_outlier_colors = np.zeros((num_clip_outliers, 3))
                            
                            min_clip_sim = clip_outlier_similarities.min()
                            max_clip_sim = clip_outlier_similarities.max()
                            
                            for i, sim in enumerate(clip_outlier_similarities):
                                if max_clip_sim > min_clip_sim:
                                    intensity = 0.3 + 0.7 * (sim - min_clip_sim) / (max_clip_sim - min_clip_sim)
                                else:
                                    intensity = 1.0
                                # Blue color scheme for CLIP outliers
                                clip_outlier_colors[i] = [0.2 * intensity, 0.4 * intensity, 1.0 * intensity]
                            
                            # Log CLIP semantic outliers
                            rr.log("world/clip_semantic_outliers", 
                                   rr.Points3D(clip_outlier_points, colors=clip_outlier_colors, radii=0.025))
                            
                            print(f"🔍 Also showing {len(clip_outlier_points)} original CLIP semantic outliers (blue)")
                else:
                    # Clear CLIP outliers if not using hybrid approach
                    rr.log("world/clip_semantic_outliers", rr.Clear())
                
                # Create mesh for highlighted points if mesh creation is enabled
                if self.create_mesh and HAS_OPEN3D and len(highlight_points) > 100:
                    print(f"🔺 Creating mesh for {len(highlight_points)} highlighted points...")
                    try:
                        highlight_mesh_vertices, highlight_mesh_faces, highlight_mesh_colors = estimate_normals_and_mesh(
                            highlight_points, highlight_colors, method='ball_pivoting'
                        )
                        
                        if highlight_mesh_vertices is not None and highlight_mesh_faces is not None:
                            if highlight_mesh_colors is not None:
                                rr.log("world/highlighted_mesh", 
                                       rr.Mesh3D(vertex_positions=highlight_mesh_vertices, 
                                               triangle_indices=highlight_mesh_faces,
                                               vertex_colors=highlight_mesh_colors))
                            else:
                                rr.log("world/highlighted_mesh", 
                                       rr.Mesh3D(vertex_positions=highlight_mesh_vertices, 
                                               triangle_indices=highlight_mesh_faces))
                            print(f"✅ Highlighted mesh created: {len(highlight_mesh_vertices)} vertices, {len(highlight_mesh_faces)} faces")
                    except Exception as e:
                        print(f"⚠️  Could not create highlighted mesh: {e}")
                
                # Log comprehensive search statistics with outlier info
                rr.log("stats/search/current_query", rr.TextLog(f"Query: '{query}'", level=rr.TextLogLevel.INFO))
                rr.log("stats/search/num_results", rr.Scalars(len(highlight_points)))
                rr.log("stats/search/top_similarity", rr.Scalars(float(similarities.max())))
                rr.log("stats/search/mean_similarity", rr.Scalars(float(similarities.mean())))
                rr.log("stats/search/threshold", rr.Scalars(display_threshold))
                rr.log("stats/search/detection_method", rr.TextLog(method_used))
                
                if use_statistical_outliers and self.last_outlier_stats:
                    # Log detailed statistical outlier information
                    stats = self.last_outlier_stats
                    rr.log("stats/outliers/mean_similarity", rr.Scalars(float(stats.get('mean', 0))))
                    rr.log("stats/outliers/median_similarity", rr.Scalars(float(stats.get('median', 0))))
                    rr.log("stats/outliers/std_similarity", rr.Scalars(float(stats.get('std', 0))))
                    rr.log("stats/outliers/iqr", rr.Scalars(float(stats.get('iqr', 0))))
                    rr.log("stats/outliers/q25", rr.Scalars(float(stats.get('q25', 0))))
                    rr.log("stats/outliers/q75", rr.Scalars(float(stats.get('q75', 0))))
                    rr.log("stats/outliers/outlier_percentage", rr.Scalars(float(stats.get('outlier_percentage', 0))))
                    
                    # Log DINO filtering info if used
                    if stats.get('dino_filtering_enabled', False):
                        rr.log("stats/outliers/dino_enabled", rr.Scalars(1))
                        rr.log("stats/outliers/points_before_dino", rr.Scalars(float(stats.get('points_before_dino', 0))))
                        rr.log("stats/outliers/points_after_dino", rr.Scalars(float(stats.get('points_after_dino', 0))))
                        dino_reduction = ((stats.get('points_before_dino', 0) - stats.get('points_after_dino', 0)) / max(stats.get('points_before_dino', 1), 1)) * 100
                        rr.log("stats/outliers/dino_reduction_percent", rr.Scalars(float(dino_reduction)))
                    else:
                        rr.log("stats/outliers/dino_enabled", rr.Scalars(0))
                    
                    # Create detailed statistical summary with DINO info
                    dino_info = ""
                    if stats.get('dino_filtering_enabled', False):
                        dino_cluster_info = stats.get('dino_cluster_info', {})
                        points_before = stats.get('points_before_dino', 0)
                        points_after = stats.get('points_after_dino', 0)
                        reduction_pct = ((points_before - points_after) / max(points_before, 1)) * 100
                        
                        dino_info = f"""
🦕 DINO Structural Filtering:
   • Input: {points_before:,} CLIP outliers → Output: {points_after:,} structured points
   • Reduction: {reduction_pct:.1f}% scattered points filtered out
   • Method: {dino_cluster_info.get('method', 'unknown')} clustering
   • Clusters kept: {dino_cluster_info.get('clusters_kept', 'unknown')}"""
                    else:
                        reason = stats.get('reason', 'unknown')
                        dino_info = f"\n🦕 DINO Filtering: Disabled ({reason})"
                    
                    stats_summary = f"""📊 HYBRID CLIP+DINO ANALYSIS: '{query}'
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🎯 Final Results: {len(highlight_points):,} / {len(self.points):,} points ({stats.get('outlier_percentage', 0):.1f}%)
📈 CLIP Outlier Method: {method_used}
📊 CLIP Threshold: {display_threshold:.3f} (automatically determined){dino_info}

🎨 Visualization:
   • Blue points: {points_before:,} CLIP semantic outliers (all matches)
   • Gold points: {points_after:,} DINO filtered results (structured matches)

📋 Similarity Distribution:
   • Mean: {stats.get('mean', 0):.3f} | Median: {stats.get('median', 0):.3f} | Std: {stats.get('std', 0):.3f}
   • Q25: {stats.get('q25', 0):.3f} | Q75: {stats.get('q75', 0):.3f} | IQR: {stats.get('iqr', 0):.3f}
   • Range: {stats.get('min', 0):.3f} - {stats.get('max', 0):.3f}

🎨 Highlighted Range: {similarities[highlight_indices].min():.3f} - {similarities[highlight_indices].max():.3f}

💡 Hybrid CLIP+DINO combines semantic understanding with structural coherence
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"""
                    
                    rr.log("docs/search_summary", rr.TextLog(stats_summary, level=rr.TextLogLevel.INFO))
                
                else:
                    # Traditional method summary
                    traditional_summary = f"""🎯 TRADITIONAL THRESHOLD SEARCH: '{query}'
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🎯 Results: {len(highlight_points):,} / {len(self.points):,} points ({len(highlight_points)/len(self.points)*100:.1f}%)
📊 Fixed threshold: {threshold:.3f} | Top-k limit: {top_k}
📈 Similarity range: {similarities[highlight_indices].min():.3f} - {similarities[highlight_indices].max():.3f}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"""
                    
                    rr.log("docs/search_summary", rr.TextLog(traditional_summary, level=rr.TextLogLevel.INFO))
                
                # Console output
                if use_statistical_outliers:
                    if use_dino_filtering and self.has_dino_features and self.last_outlier_stats.get('dino_filtering_enabled', False):
                        dino_info = self.last_outlier_stats.get('dino_cluster_info', {})
                        points_before = self.last_outlier_stats.get('points_before_dino', 0)
                        points_after = self.last_outlier_stats.get('points_after_dino', 0)
                        print(f"✨ Hybrid CLIP+DINO search results for '{query}':")
                        print(f"   CLIP outliers: {points_before} → DINO filtered: {points_after}")
                        print(f"   Method: {method_used}, Threshold: {display_threshold:.3f}")
                        print(f"   Final results: {len(highlight_points)} ({self.last_outlier_stats.get('outlier_percentage', 0):.1f}%)")
                    else:
                        print(f"✨ Statistical outlier search results for '{query}':")
                        print(f"   Method: {method_used}, Threshold: {display_threshold:.3f}")
                        print(f"   Outliers: {len(highlight_points)} ({self.last_outlier_stats.get('outlier_percentage', 0):.1f}%)")
                else:
                    print(f"✨ Traditional search results for '{query}':")
                    print(f"   Fixed threshold: {threshold:.3f}, Results: {len(highlight_points)}")
                
            else:
                # No matches found
                rr.log("world/text_similarity_highlights", rr.Clear())
                rr.log("world/clip_semantic_outliers", rr.Clear())
                rr.log("world/highlighted_mesh", rr.Clear())
                
                # Log no results stats
                rr.log("stats/search/current_query", rr.TextLog(f"Query: '{query}' (NO MATCHES)", level=rr.TextLogLevel.WARN))
                rr.log("stats/search/num_results", rr.Scalars(0))
                rr.log("stats/search/top_similarity", rr.Scalars(float(similarities.max()) if len(similarities) > 0 else 0.0))
                rr.log("stats/search/mean_similarity", rr.Scalars(float(similarities.mean()) if len(similarities) > 0 else 0.0))
                rr.log("stats/search/threshold", rr.Scalars(display_threshold))
                rr.log("stats/search/detection_method", rr.TextLog(method_used))
                
                method_desc = "statistical" if use_statistical_outliers else "traditional"
                no_match_summary = f"""⚠️  NO MATCHES FOR: '{query}' ({method_desc} method)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🎯 Results: 0 / {len(self.points):,} points 
📊 Max similarity: {similarities.max():.3f}
📈 Detection method: {method_used}
💡 Try: different query terms or switch detection method
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"""
                
                rr.log("docs/search_summary", rr.TextLog(no_match_summary, level=rr.TextLogLevel.WARN))
                
                print(f"⚠️  No matches found for '{query}' using {method_desc} method")
                print(f"   Max similarity: {similarities.max():.3f}, Method: {method_used}")
                
        except Exception as e:
            print(f"❌ Error processing query '{query}': {e}")
            rr.log("errors/search", rr.TextLog(f"Error: {str(e)}", level=rr.TextLogLevel.ERROR))
    
    def run_interactive_session(self, port=9878):
        """Run the main interactive session."""
        # Initialize Rerun
        rr.init("Interactive_Text_Search", spawn=False)
        rr.serve_grpc(grpc_port=port)
        print(f"🌐 Rerun server started on port {port}")
        
        # Setup coordinate frame
        rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Y_UP, static=True)
        rr.set_time_seconds("timeline", 0.0)
        
        # Log the base pointcloud
        print("📊 Logging base pointcloud...")
        rr.log("world/pointcloud_rgb", 
               rr.Points3D(self.points, colors=self.rgb, radii=0.008), 
               static=True)
        
        # Log base mesh if available
        if self.base_mesh_vertices is not None and self.base_mesh_faces is not None:
            print("🔺 Logging base mesh...")
            if self.base_mesh_colors is not None:
                rr.log("world/base_mesh", 
                       rr.Mesh3D(vertex_positions=self.base_mesh_vertices, 
                               triangle_indices=self.base_mesh_faces,
                               vertex_colors=self.base_mesh_colors), 
                       static=True)
            else:
                rr.log("world/base_mesh", 
                       rr.Mesh3D(vertex_positions=self.base_mesh_vertices, 
                               triangle_indices=self.base_mesh_faces), 
                       static=True)
            print(f"✅ Base mesh logged: {len(self.base_mesh_vertices)} vertices, {len(self.base_mesh_faces)} faces")
        
        # Log CLIP feature visualization
        print("🎨 Logging CLIP feature visualization...")
        clip_colors = features_to_colors_pca(self.features_info['clip'], method='hsv')
        rr.log("world/pointcloud_clip_features", 
               rr.Points3D(self.points, colors=clip_colors, radii=0.008), 
               static=True)
        
        # Log comprehensive initial stats with better organization
        bbox_min = self.points.min(axis=0)
        bbox_max = self.points.max(axis=0)
        bbox_size = bbox_max - bbox_min
        
        rr.log("stats/pointcloud/total_points", rr.Scalars(len(self.points)), static=True)
        rr.log("stats/pointcloud/clip_feature_dim", rr.Scalars(self.features_info['clip'].shape[1]), static=True)
        rr.log("stats/pointcloud/bbox_size_x", rr.Scalars(float(bbox_size[0])), static=True)
        rr.log("stats/pointcloud/bbox_size_y", rr.Scalars(float(bbox_size[1])), static=True)
        rr.log("stats/pointcloud/bbox_size_z", rr.Scalars(float(bbox_size[2])), static=True)
        rr.log("stats/pointcloud/bbox_volume", rr.Scalars(float(np.prod(bbox_size))), static=True)
        
        if self.base_mesh_vertices is not None:
            rr.log("stats/pointcloud/mesh_vertices", rr.Scalars(len(self.base_mesh_vertices)), static=True)
            rr.log("stats/pointcloud/mesh_faces", rr.Scalars(len(self.base_mesh_faces)), static=True)
        
        # Create a comprehensive pointcloud summary
        mesh_info = ""
        if self.base_mesh_vertices is not None:
            mesh_info = f"""
🔺 Base Mesh: {len(self.base_mesh_vertices):,} vertices, {len(self.base_mesh_faces):,} faces"""
        
        pointcloud_summary = f"""📊 INTERACTIVE TEXT SEARCH READY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
☁️  Pointcloud: {len(self.points):,} points
🧠 Features: {self.features_info['clip'].shape[1]}D CLIP features{' + DINO features' if self.has_dino_features else ''}
📦 Bounding Box: [{bbox_size[0]:.2f} × {bbox_size[1]:.2f} × {bbox_size[2]:.2f}] units{mesh_info}
🎯 Ready for interactive text search!

💡 How to search:
   • Type queries in terminal: "chair", "red table", "wooden furniture"
   • Edit query.txt file in any text editor
   • Adjust settings: "threshold=0.15", "topk=300"
   • Type "help" for more commands

🎨 Visualization Colors:
   • Blue points: CLIP semantic matches (raw semantic similarity)
   • Gold points: DINO filtered results (structured, coherent matches)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"""
        
        rr.log("docs/instructions", rr.TextLog(pointcloud_summary, level=rr.TextLogLevel.INFO), static=True)
        
        # Initialize search stats structure
        rr.log("stats/search/current_query", rr.TextLog("No active search", level=rr.TextLogLevel.INFO), static=True)
        rr.log("stats/search/num_results", rr.Scalars(0), static=True)
        rr.log("stats/search/threshold", rr.Scalars(0.2), static=True)
        rr.log("stats/search/top_k", rr.Scalars(200), static=True)
        
        # Start watchers and input
        file_observer = self.start_file_watcher()
        terminal_thread = self.start_terminal_input()
        
        # Settings
        current_threshold = 0.2
        current_top_k = 200
        current_outlier_method = self.outlier_method
        current_use_statistical_outliers = self.use_statistical_outliers
        current_use_dino_filtering = self.use_dino_filtering
        
        try:
            # Process initial query from file
            if self.query_file.exists():
                with open(self.query_file, 'r') as f:
                    initial_query = f.read().strip()
                if initial_query:
                    print(f"🔍 Processing initial query: '{initial_query}'")
                    self.process_text_query(initial_query, current_top_k, current_threshold, current_outlier_method, current_use_statistical_outliers, current_use_dino_filtering)
            
            # Main processing loop
            print("🚀 Interactive session started! Use Rerun viewer to see results.")
            
            while self.running:
                try:
                    # Check for new queries (with timeout)
                    source, query = self.query_queue.get(timeout=0.5)
                    
                    if source == "threshold":
                        current_threshold = query
                        print(f"🎛️  Updated threshold to: {current_threshold}")
                        # Re-process current query with new threshold
                        if self.current_query:
                            self.process_text_query(self.current_query, current_top_k, current_threshold, current_outlier_method, current_use_statistical_outliers, current_use_dino_filtering)
                    
                    elif source == "topk":
                        current_top_k = query
                        print(f"🎛️  Updated top-k to: {current_top_k}")
                        # Re-process current query with new top-k
                        if self.current_query:
                            self.process_text_query(self.current_query, current_top_k, current_threshold, current_outlier_method, current_use_statistical_outliers, current_use_dino_filtering)
                    
                    elif source == "outlier_method":
                        current_outlier_method = query
                        print(f"🎛️  Updated outlier detection method to: {current_outlier_method}")
                        # Re-process current query with new outlier method
                        if self.current_query:
                            self.process_text_query(self.current_query, current_top_k, current_threshold, current_outlier_method, current_use_statistical_outliers, current_use_dino_filtering)
                    
                    elif source == "use_statistical_outliers":
                        current_use_statistical_outliers = query.lower() == 'true'
                        print(f"🎛️  Statistical outlier detection: {current_use_statistical_outliers}")
                        # Re-process current query with new outlier method
                        if self.current_query:
                            self.process_text_query(self.current_query, current_top_k, current_threshold, current_outlier_method, current_use_statistical_outliers, current_use_dino_filtering)
                    
                    elif source == "use_dino_filtering":
                        current_use_dino_filtering = query.lower() == 'true'
                        print(f"🎛️  DINO structural filtering: {current_use_dino_filtering}")
                        # Re-process current query with new DINO filtering setting
                        if self.current_query:
                            self.process_text_query(self.current_query, current_top_k, current_threshold, current_outlier_method, current_use_statistical_outliers, current_use_dino_filtering)
                    
                    else:
                        # Regular text query
                        self.current_query = query
                        self.process_text_query(query, current_top_k, current_threshold, current_outlier_method, current_use_statistical_outliers, current_use_dino_filtering)
                        
                        # Update query file if query came from terminal
                        if source == "terminal" and query:
                            with open(self.query_file, 'w') as f:
                                f.write(query + '\n')
                    
                except queue.Empty:
                    continue
                except KeyboardInterrupt:
                    break
                    
        except KeyboardInterrupt:
            print("\n⏹️  Interactive session interrupted")
        finally:
            self.running = False
            file_observer.stop()
            file_observer.join()
            print("🛑 Interactive session ended")

def main():
    parser = argparse.ArgumentParser(description="Interactive Text Search for Featurized Pointclouds with Hybrid CLIP+DINO Filtering")
    parser.add_argument("pointcloud_dir", help="Directory containing featurized pointcloud .pt files")
    parser.add_argument("--file-type", default="combined", help="Which .pt file to use")
    parser.add_argument("--port", type=int, default=9878, help="Rerun server port")
    parser.add_argument("--clip-model", default="ViT-B/32", choices=["ViT-B/32", "ViT-L/14"], 
                       help="CLIP model version")
    parser.add_argument("--create-mesh", action="store_true", default=True,
                       help="Create ball_pivoting mesh for better visualization (default: True)")
    parser.add_argument("--no-mesh", action="store_true",
                       help="Disable mesh creation for faster loading")
    
    # Statistical outlier detection options
    parser.add_argument("--outlier-method", default="adaptive", 
                       choices=["iqr", "percentile", "z_score", "adaptive", "combined"],
                       help="Statistical outlier detection method (default: adaptive)")
    parser.add_argument("--use-traditional", action="store_true",
                       help="Use traditional fixed threshold method instead of statistical outliers")
    
    # DINO filtering options
    parser.add_argument("--no-dino-filtering", action="store_true",
                       help="Disable DINO structural filtering (use CLIP-only)")
    
    args = parser.parse_args()
    
    # Handle mesh creation flag
    create_mesh = args.create_mesh and not args.no_mesh
    
    # Handle outlier detection settings
    use_statistical_outliers = not args.use_traditional
    
    # Handle DINO filtering setting
    use_dino_filtering = not args.no_dino_filtering
    
    # Discover files
    try:
        files_info = discover_featurized_files(args.pointcloud_dir)
        print("📁 Available files:")
        for key, path in files_info.items():
            print(f"   {key}: {os.path.basename(path)}")
    except Exception as e:
        print(f"❌ Error discovering files: {e}")
        return
    
    # Validate file selection
    if args.file_type not in files_info:
        print(f"❌ File type '{args.file_type}' not found.")
        print(f"Available options: {list(files_info.keys())}")
        return
    
    # Check CLIP encoder availability
    if not HAS_CLIP_ENCODER:
        print("❌ CLIP encoder not available. This script requires CLIP functionality.")
        return
    
    # Check mesh creation capability
    if create_mesh and not HAS_OPEN3D:
        print("⚠️  Open3D not available. Disabling mesh creation.")
        create_mesh = False
    
    # Start interactive session
    try:
        interactive_search = InteractiveTextSearch(
            files_info, 
            args.file_type, 
            args.clip_model,
            create_mesh=create_mesh,
            outlier_method=args.outlier_method,
            use_statistical_outliers=use_statistical_outliers,
            use_dino_filtering=use_dino_filtering
        )
        
        print("\n" + "="*80)
        print("🎯 ENHANCED INTERACTIVE TEXT SEARCH WITH HYBRID CLIP+DINO FILTERING")
        print("="*80)
        
        if use_dino_filtering:
            print("🔗 Hybrid CLIP+DINO filtering enabled")
            print("💡 Benefits:")
            print("   • CLIP: Semantic understanding (finds 'chair', 'red table')")
            print("   • DINO: Structural coherence (filters scattered points)")
            print("   • Result: Clean object boundaries instead of noise")
            print("   • Visualization: Blue points (CLIP semantic) + Gold points (DINO filtered)")
        elif use_statistical_outliers:
            print("📈 Statistical CLIP-only outlier detection enabled")
            print(f"   Method: {args.outlier_method}")
            print("💡 Benefits:")
            print("   • Automatically adapts to object size (apple vs Christmas tree)")
            print("   • No manual threshold tuning required")
            print("   • More robust results across different queries")
        else:
            print("🎯 Traditional fixed threshold method enabled")
            print("💡 Use statistical or hybrid mode for better results")
        
        if create_mesh:
            print("🔺 Mesh creation enabled - ball_pivoting method for better visualization")
        else:
            print("📊 Point cloud only mode - faster loading, no mesh")
        
        print("\n🎮 Interactive Commands:")
        print("   • Type queries: 'chair', 'red table', 'wooden furniture'")
        if use_dino_filtering:
            print("   • Toggle methods: 'use_dino_filtering=false' (CLIP-only)")
        print("   • Switch outlier detection: 'outlier_method=iqr'")
        print("   • Traditional settings: 'threshold=0.15', 'topk=300'")
        print("   • Help: 'help'")
        print("="*80 + "\n")
            
        interactive_search.run_interactive_session(args.port)
    except Exception as e:
        print(f"❌ Error starting interactive session: {e}")
        raise

if __name__ == "__main__":
    main() 