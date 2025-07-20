#!/usr/bin/env python3
"""
Interactive Text Search for Featurized Pointclouds using Rerun SDK

This script provides real-time interactive text search capabilities:
1. Terminal input: Type queries directly in the terminal
2. Socket interface: Send queries via network socket

Usage:
  python visualize_interactive_text_search.py data/ARKitScenes_fpt --file-type 42447230

Controls:
  - Type in terminal for immediate search
  - Press 'q' + Enter to quit
  - Press 'clear' + Enter to clear highlights
"""
import queue
import select
import sys
import threading
from pathlib import Path

import numpy as np
import rerun as rr
import time  # Add this import at the top of the file
import torch
import torch.nn.functional as F
from sklearn.cluster import DBSCAN

from src.clip_encoder import ClipEncoder
from src.visualize_featurized_pointcloud import (
    create_text_similarity_highlights,
    estimate_normals_and_mesh,
    features_to_colors_pca,
    load_featurized_pointcloud,
    load_original_pointcloud,
)

def detect_similarity_outliers_enhanced(similarities, dino_features=None, points=None, method='adaptive', 
                                       min_threshold=0.1, percentile_threshold=95, iqr_multiplier=2.5, 
                                       z_score_threshold=2.0, min_points=5, dino_weight=0.3):
    """
    Enhanced outlier detection that leverages DINO features for better structural coherence.
    Step 1: Finds 1,000 initial points with high CLIP similarity
    Step 2: Analyzes DINO features to find structural patterns
    Step 3: Checks 3D spatial relationships
    Step 4: Combines semantic + structural scores
    Step 5: Returns ~250 points (top 25%) that form coherent structures

    Args:
        similarities: numpy array of similarity scores
        dino_features: numpy array [N, D] - DINO features (optional)
        points: numpy array [N, 3] - 3D point coordinates (optional)
        method: 'iqr', 'percentile', 'z_score', 'adaptive', 'combined', or 'enhanced_adaptive'
        min_threshold: minimum similarity to consider (filters noise)
        percentile_threshold: percentile threshold for percentile method
        iqr_multiplier: multiplier for IQR method
        z_score_threshold: threshold for z-score method
        min_points: minimum number of points to return
        dino_weight: weight for DINO structural coherence (0-1)
    
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
    
    if method == 'enhanced_adaptive':
        # Enhanced adaptive method that considers DINO structural coherence
        if dino_features is not None and points is not None and len(valid_indices) > min_points:
            # Get initial outliers using standard adaptive method
            initial_outliers, _, _, _ = detect_similarity_outliers(
                similarities, method='adaptive', min_threshold=min_threshold,
                percentile_threshold=percentile_threshold, iqr_multiplier=iqr_multiplier,
                z_score_threshold=z_score_threshold, min_points=min_points
            )
            
            if len(initial_outliers) > 0:
                # Calculate DINO structural coherence for initial outliers
                outlier_dino_features = dino_features[initial_outliers]
                outlier_points = points[initial_outliers]
                
                # Normalize DINO features
                dino_norms = np.linalg.norm(outlier_dino_features, axis=1, keepdims=True)
                dino_norms[dino_norms == 0] = 1e-8
                normalized_dino = outlier_dino_features / dino_norms
                
                # Calculate DINO feature similarity matrix
                dino_similarities = np.dot(normalized_dino, normalized_dino.T)
                
                # Calculate spatial proximity matrix (Closer points = higher similarity)
                from scipy.spatial.distance import cdist
                spatial_distances = cdist(outlier_points, outlier_points)
                spatial_similarities = 1.0 / (1.0 + spatial_distances)
                
                # Combine DINO and spatial coherence (default: 70% dino, 30% spatial)
                structural_coherence = (dino_weight * dino_similarities + 
                                      (1 - dino_weight) * spatial_similarities)
                
                # Calculate average structural coherence for each point
                avg_coherence = np.mean(structural_coherence, axis=1)
                
                # Get similarity scores for outliers
                outlier_similarities = similarities[initial_outliers]
                 
                # Combine semantic similarity with structural coherence (default: 70% clip, 30% dino)
                combined_scores = (0.7 * outlier_similarities + 0.3 * avg_coherence)
                
                # Find points with high combined scores
                coherence_threshold = np.percentile(combined_scores, 75)  # Top 25% by combined score
                high_coherence_mask = combined_scores >= coherence_threshold
                
                final_outlier_indices = initial_outliers[high_coherence_mask]
                
                stats.update({
                    'enhanced_adaptive': True,
                    'initial_outliers': len(initial_outliers),
                    'final_outliers': len(final_outlier_indices),
                    'coherence_threshold': coherence_threshold,
                    'dino_weight': dino_weight,
                    'avg_coherence_range': [avg_coherence.min(), avg_coherence.max()],
                    'combined_score_range': [combined_scores.min(), combined_scores.max()]
                })
                
                if len(final_outlier_indices) >= min_points:
                    threshold = similarities[final_outlier_indices].min()
                    method_used = 'enhanced_adaptive'
                    return final_outlier_indices, threshold, method_used, stats
        
        # Fall back to standard adaptive if DINO features not available or insufficient results
        return detect_similarity_outliers(
            similarities, method='adaptive', min_threshold=min_threshold,
            percentile_threshold=percentile_threshold, iqr_multiplier=iqr_multiplier,
            z_score_threshold=z_score_threshold, min_points=min_points
        )
    
    else:
        # Use standard outlier detection methods
        return detect_similarity_outliers(
            similarities, method=method, min_threshold=min_threshold,
            percentile_threshold=percentile_threshold, iqr_multiplier=iqr_multiplier,
            z_score_threshold=z_score_threshold, min_points=min_points
        )

def detect_similarity_outliers(similarities, method='adaptive', min_threshold=0.1, 
                              percentile_threshold=95, iqr_multiplier=2.5, 
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
        stats['iqr_multiplier'] = iqr_multiplier
        
    elif method == 'percentile':
        # Percentile-based detection
        threshold = np.percentile(valid_similarities, percentile_threshold)
        outlier_mask = valid_similarities >= threshold
        method_used = 'percentile'
        stats['percentile_threshold'] = percentile_threshold
        
    elif method == 'z_score':
        # Z-score based detection
        z_scores = (valid_similarities - mean_sim) / (std_sim + 1e-8)
        threshold = mean_sim + z_score_threshold * std_sim
        outlier_mask = z_scores >= z_score_threshold
        method_used = 'z_score'
        stats['z_score_threshold'] = z_score_threshold
        
    elif method == 'adaptive':
        # Adaptive method: choose based on data characteristics
        if iqr > 0.05:  # If there's good spread, use IQR
            threshold = q75 + iqr_multiplier * iqr
            outlier_mask = valid_similarities >= threshold
            method_used = 'iqr'
            stats['iqr_multiplier'] = iqr_multiplier
        elif std_sim > 0.02:  # If there's decent variation, use z-score
            z_scores = (valid_similarities - mean_sim) / (std_sim + 1e-8)
            threshold = mean_sim + z_score_threshold * std_sim
            outlier_mask = z_scores >= z_score_threshold
            method_used = 'z_score'
            stats['z_score_threshold'] = z_score_threshold
        else:  # Fall back to percentile
            threshold = np.percentile(valid_similarities, percentile_threshold)
            outlier_mask = valid_similarities >= threshold
            method_used = 'percentile'
            stats['percentile_threshold'] = percentile_threshold
            
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
        stats['iqr_multiplier'] = iqr_multiplier
        stats['z_score_threshold'] = z_score_threshold
        stats['percentile_threshold'] = percentile_threshold
    
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
                                                 iqr_multiplier=2.5,
                                                 z_score_threshold=2.0,
                                                 min_points=5,
                                                 highlight_color=[1.0, 0.8, 0.0]):
    """
    Create highlights based on statistical outlier detection of CLIP similarities.
    
    This is an enhanced version that uses statistical methods to automatically 
    identify the high-similarity outlier group instead of fixed thresholds.
    """
    
    if clip_encoder is None:
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
    outlier_indices, threshold_used, method_used, outlier_stats = detect_similarity_outliers_enhanced(
        similarities, 
        dino_features=None,
        points=None,
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
        n_clusters_int = min(max(2, len(indices) // 20), 8)
    else:
        n_clusters_int = int(n_clusters)
    
    try:
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score
        
        # Try different cluster numbers and pick best
        best_clusters = 2
        best_score = -1
        
        for k in range(2, min(n_clusters_int + 1, len(indices) // min_cluster_size + 1)):
            if k >= len(indices):
                break
                
            kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto")
            cluster_labels = kmeans.fit_predict(combined_features)
            
            if len(np.unique(cluster_labels)) > 1:
                score = silhouette_score(combined_features, cluster_labels)
                if score > best_score:
                    best_score = score
                    best_clusters = k
        
        # Final clustering with best number of clusters
        kmeans = KMeans(n_clusters=best_clusters, random_state=42, n_init="auto")
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
    
    if clip_encoder is None:
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
    clip_outlier_indices, threshold_used, method_used, outlier_stats = detect_similarity_outliers_enhanced(
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

def run_all_outlier_methods(points, clip_features, text_query, clip_encoder=None, 
                           dino_features=None, min_threshold=0.1, percentile_threshold=95, iqr_multiplier=2.5, 
                           z_score_threshold=2.0, min_points=5):
    """
    Run all outlier detection methods simultaneously and return results for comparison.
    
    Args:
        points: numpy array [N, 3] - 3D point coordinates
        clip_features: numpy array [N, D] - CLIP features for all points
        text_query: str - text query to search for
        clip_encoder: CLIP encoder instance
        dino_features: numpy array [N, D] - DINO features (optional)
        min_threshold: minimum similarity to consider
        percentile_threshold: percentile threshold for percentile method
        iqr_multiplier: multiplier for IQR method
        z_score_threshold: threshold for z-score method
        min_points: minimum number of points to return
    
    Returns:
        results: dictionary with results for each method
        similarities: numpy array of all similarity scores
    """
    
    if clip_encoder is None:
        print("⚠️  CLIP encoder not available for text similarity highlighting")
        return {}, np.array([])
    
    if clip_features is None or len(clip_features) == 0:
        print("⚠️  No CLIP features available for text similarity highlighting")
        return {}, np.array([])
    
    print(f"🔍 Computing ALL outlier methods for query: '{text_query}'")
    
    # Encode the text query
    try:
        text_features = clip_encoder.encode_text(text_query)
        text_features = text_features.cpu().numpy()
        text_features = text_features / np.linalg.norm(text_features, axis=1, keepdims=True)
        print(f"✓ Text encoded to {text_features.shape}")
        
    except Exception as e:
        print(f"❌ Error encoding text query: {e}")
        return {}, np.array([])
    
    # Check dimension compatibility
    clip_feature_dim = clip_features.shape[1]
    text_feature_dim = text_features.shape[1]
    
    if clip_feature_dim != text_feature_dim:
        print(f"❌ Dimension mismatch: CLIP features ({clip_feature_dim}D) vs Text features ({text_feature_dim}D)")
        return {}, np.array([])
    
    # Normalize CLIP features and compute similarities
    norms = np.linalg.norm(clip_features, axis=1, keepdims=True)
    zero_mask = norms[:, 0] == 0
    if np.any(zero_mask):
        print(f"⚠️  Found {np.sum(zero_mask)} zero feature vectors, setting to small values")
        norms[zero_mask] = 1e-8
    
    clip_features_norm = clip_features / norms
    similarities = np.dot(clip_features_norm, text_features.T).flatten()
    
    print(f"📈 Similarity stats - Min: {similarities.min():.3f}, Max: {similarities.max():.3f}, Mean: {similarities.mean():.3f}")
    
    # Define all methods to test
    methods = ['percentile', 'z_score', 'adaptive', 'enhanced_adaptive', 'combined', 'expansion', 'dino_clip_smoothing']
    
    # Color scheme for different methods
    method_colors = {
        'percentile': [0.0, 1.0, 0.0],  # Green
        'z_score': [0.0, 0.0, 1.0],     # Blue
        'adaptive': [1.0, 1.0, 0.0],    # Yellow
        'enhanced_adaptive': [1.0, 0.5, 0.0],  # Orange
        'combined': [1.0, 0.0, 1.0],    # Magenta
        'topk': [1.0, 0.0, 0.0],        # Red for topk
        'expansion': [0.0, 1.0, 1.0],   # Cyan for expansion
        'dino_clip_smoothing': [1.0, 0.2, 0.2], # Bright red
    }
    
    results = {}
    
    # Run each method
    for method in methods:
        print(f"🔬 Testing {method} method...")
        if method == 'expansion':
            if dino_features is not None:
                expansion_indices = expand_cluster_from_top_clip_points(
                    points, clip_features, dino_features, similarities,
                    top_n=50, max_expansion=2000, dino_weight=0.6, spatial_weight=0.4, dino_thresh=0.6, spatial_thresh=0.18, clip_thresh=0.18
                )
                highlight_points = points[expansion_indices]
                highlight_similarities = similarities[expansion_indices]
                num_highlights = len(expansion_indices)
                highlight_colors = np.zeros((num_highlights, 3))
                min_sim = highlight_similarities.min() if num_highlights > 0 else 0
                max_sim = highlight_similarities.max() if num_highlights > 0 else 1
                base_color = method_colors['expansion']
                for i, sim in enumerate(highlight_similarities):
                    if max_sim > min_sim:
                        intensity = 0.4 + 0.6 * (sim - min_sim) / (max_sim - min_sim)
                    else:
                        intensity = 1.0
                    highlight_colors[i] = [c * intensity for c in base_color]
                results['expansion'] = {
                    'indices': expansion_indices,
                    'points': highlight_points,
                    'colors': highlight_colors,
                    'similarities': highlight_similarities,
                    'threshold': None,
                    'method_used': 'expansion',
                    'stats': {'num_points': num_highlights, 'top_n': 50, 'max_expansion': 2000}
                }
                print(f"   ✓ expansion: {num_highlights} points (top 50 CLIP, expanded)")
            else:
                results['expansion'] = {
                    'indices': np.array([]),
                    'points': np.array([]).reshape(0, 3),
                    'colors': np.array([]).reshape(0, 3),
                    'similarities': np.array([]),
                    'threshold': None,
                    'method_used': 'expansion',
                    'stats': {'num_points': 0, 'reason': 'no_dino_features'}
                }
                print(f"   ⚠️  expansion: DINO features not available")
            continue
        elif method == 'dino_clip_smoothing':
            if dino_features is not None:
                # Use the first text feature vector (shape [1, D])
                indices, highlight_points, highlight_colors, final_sim_np, threshold_used, method_used, stats = dino_guided_clip_smoothing(
                    points, clip_features, dino_features, text_features, threshold=0.22, k=10, mix_alpha=0.5)
                results['dino_clip_smoothing'] = {
                    'indices': indices,
                    'points': highlight_points,
                    'colors': highlight_colors,
                    'similarities': final_sim_np,  # <-- Use the returned similarities
                    'threshold': threshold_used,
                    'method_used': method_used,
                    'stats': stats
                }
                print(f"   ✓ dino_clip_smoothing: {len(indices)} points, threshold: {threshold_used}")
            else:
                results['dino_clip_smoothing'] = {
                    'indices': np.array([]),
                    'points': np.array([]).reshape(0, 3),
                    'colors': np.array([]).reshape(0, 3),
                    'similarities': np.array([]),
                    'threshold': 0.0,
                    'method_used': 'dino_clip_smoothing',
                    'stats': {'num_points': 0, 'reason': 'no_dino_features'}
                }
                print(f"   ⚠️  dino_clip_smoothing: DINO features not available")
            continue
        outlier_indices, threshold_used, method_used, outlier_stats = detect_similarity_outliers_enhanced(
            similarities, 
            dino_features=dino_features,
            points=points,
            method=method,
            min_threshold=min_threshold,
            percentile_threshold=percentile_threshold,
            iqr_multiplier=iqr_multiplier,
            z_score_threshold=z_score_threshold,
            min_points=min_points
        )
        if len(outlier_indices) > 0:
            highlight_points = points[outlier_indices]
            highlight_similarities = similarities[outlier_indices]
            
            # Create gradient colors based on similarity strength
            num_highlights = len(outlier_indices)
            highlight_colors = np.zeros((num_highlights, 3))
            
            min_outlier_sim = highlight_similarities.min()
            max_outlier_sim = highlight_similarities.max()
            
            base_color = method_colors[method]
            
            for i, sim in enumerate(highlight_similarities):
                if max_outlier_sim > min_outlier_sim:
                    intensity = 0.4 + 0.6 * (sim - min_outlier_sim) / (max_outlier_sim - min_outlier_sim)
                else:
                    intensity = 1.0
                highlight_colors[i] = [c * intensity for c in base_color]
            
            results[method] = {
                'indices': outlier_indices,
                'points': highlight_points,
                'colors': highlight_colors,
                'similarities': highlight_similarities,
                'threshold': threshold_used,
                'method_used': method_used,
                'stats': outlier_stats
            }
            
            print(f"   ✓ {method}: {len(outlier_indices)} points, threshold: {threshold_used:.3f}")
        else:
            print(f"   ⚠️  {method}: No outliers found")
            results[method] = {
                'indices': np.array([]),
                'points': np.array([]).reshape(0, 3),
                'colors': np.array([]).reshape(0, 3),
                'similarities': np.array([]),
                'threshold': 0.0,
                'method_used': method_used,
                'stats': outlier_stats
            }
    # Add topk method for comparison (always show top 500 by similarity, no threshold)
    print("🔬 Testing topk method...")
    if clip_encoder is not None and clip_features is not None and len(clip_features) > 0:
        top_k = 500
        if len(similarities) > 0:
            sorted_indices = np.argsort(similarities)[::-1][:top_k]
            topk_indices = sorted_indices
            topk_points = points[topk_indices]
            topk_similarities = similarities[topk_indices]
            highlight_color = method_colors['topk']
            min_sim = topk_similarities.min()
            max_sim = topk_similarities.max()
            num_highlights = len(topk_indices)
            topk_colors = np.zeros((num_highlights, 3))
            for i, sim in enumerate(topk_similarities):
                if max_sim > min_sim:
                    intensity = 0.4 + 0.6 * (sim - min_sim) / (max_sim - min_sim)
                else:
                    intensity = 1.0
                topk_colors[i] = [c * intensity for c in highlight_color]
            results['topk'] = {
                'indices': topk_indices,
                'points': topk_points,
                'colors': topk_colors,
                'similarities': topk_similarities,
                'threshold': None,
                'method_used': 'topk',
                'stats': {'top_k': top_k, 'num_points': len(topk_indices)}
            }
            print(f"   ✓ topk: {len(topk_indices)} points (top {top_k} by similarity)")
        else:
            results['topk'] = {
                'indices': np.array([]),
                'points': np.array([]).reshape(0, 3),
                'colors': np.array([]).reshape(0, 3),
                'similarities': np.array([]),
                'threshold': None,
                'method_used': 'topk',
                'stats': {'top_k': top_k, 'num_points': 0}
            }
            print(f"   ⚠️  topk: No points found (empty similarities)")
    else:
        results['topk'] = {
            'indices': np.array([]),
            'points': np.array([]).reshape(0, 3),
            'colors': np.array([]).reshape(0, 3),
            'similarities': np.array([]),
            'threshold': None,
            'method_used': 'topk',
            'stats': {'top_k': 500, 'num_points': 0}
        }
        print(f"   ⚠️  topk: CLIP encoder or features missing")
    
    if dino_features is not None:
        if method == 'expansion':
            expansion_indices = expand_cluster_from_top_clip_points(
                points, clip_features, dino_features, similarities,
                top_n=50, max_expansion=500, dino_weight=0.6, spatial_weight=0.4, dino_thresh=0.6, spatial_thresh=0.18, clip_thresh=0.18
            )
            highlight_points = points[expansion_indices]
            highlight_similarities = similarities[expansion_indices]
            num_highlights = len(expansion_indices)
            highlight_colors = np.zeros((num_highlights, 3))
            min_sim = highlight_similarities.min() if num_highlights > 0 else 0
            max_sim = highlight_similarities.max() if num_highlights > 0 else 1
            base_color = method_colors['expansion']
            for i, sim in enumerate(highlight_similarities):
                if max_sim > min_sim:
                    intensity = 0.4 + 0.6 * (sim - min_sim) / (max_sim - min_sim)
                else:
                    intensity = 1.0
                highlight_colors[i] = [c * intensity for c in base_color]
            results['expansion'] = {
                'indices': expansion_indices,
                'points': highlight_points,
                'colors': highlight_colors,
                'similarities': highlight_similarities,
                'threshold': None,
                'method_used': 'expansion',
                'stats': {'num_points': num_highlights, 'top_n': 50, 'max_expansion': 500}
            }
            print(f"   ✓ expansion: {num_highlights} points (top 50 CLIP, expanded)")
        else:
            results['expansion'] = {
                'indices': np.array([]),
                'points': np.array([]).reshape(0, 3),
                'colors': np.array([]).reshape(0, 3),
                'similarities': np.array([]),
                'threshold': None,
                'method_used': 'expansion',
                'stats': {'num_points': 0, 'reason': 'no_dino_features'}
            }
            print(f"   ⚠️  expansion: DINO features not available")
    
    return results, similarities

# --- New cluster expansion method ---
def expand_cluster_from_top_clip_points(points, clip_features, dino_features, similarities, top_n=50, max_expansion=2000, dino_weight=0.6, spatial_weight=0.4, dino_thresh=0.6, spatial_thresh=0.18, clip_thresh=0.18, k=20):
    """
    Region growing from top-N CLIP points with kNN, DINO, and CLIP coherence.
    Returns indices of the expanded cluster.
    """
    import numpy as np
    from sklearn.neighbors import NearestNeighbors
    # 1. Start with top-N CLIP points
    seed_indices = np.argsort(similarities)[-top_n:][::-1]
    if len(seed_indices) == 0:
        return np.array([])
    cluster_set = set(seed_indices.tolist())
    frontier = set(seed_indices.tolist())
    # Precompute normalized features
    dino_norm = dino_features / (np.linalg.norm(dino_features, axis=1, keepdims=True) + 1e-8)
    clip_norm = clip_features / (np.linalg.norm(clip_features, axis=1, keepdims=True) + 1e-8)
    # Build kNN graph (spatial)
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(points)
    knn_indices = nbrs.kneighbors(return_distance=False)
    # Compute cluster means for DINO and CLIP
    def cluster_mean(indices, arr):
        return arr[list(indices)].mean(axis=0)
    while len(cluster_set) < max_expansion and frontier:
        new_frontier = set()
        cluster_dino_mean = cluster_mean(cluster_set, dino_norm)
        cluster_clip_mean = cluster_mean(cluster_set, clip_norm)
        for idx in frontier:
            neighbors = knn_indices[idx]
            for n_idx in neighbors:
                if n_idx in cluster_set:
                    continue
                # DINO similarity to cluster mean
                dino_sim = np.dot(dino_norm[n_idx], cluster_dino_mean)
                # CLIP similarity to cluster mean
                clip_sim = np.dot(clip_norm[n_idx], cluster_clip_mean)
                # Spatial distance to cluster (min over cluster)
                spatial_dist = np.linalg.norm(points[n_idx] - points[list(cluster_set)], axis=1).min()
                if dino_sim > dino_thresh and clip_sim > clip_thresh and spatial_dist < spatial_thresh:
                    cluster_set.add(n_idx)
                    new_frontier.add(n_idx)
        if not new_frontier:
            break
        frontier = new_frontier
    return np.array(sorted(cluster_set))

class InteractiveTextSearch:
    """Main class for interactive text search visualization with statistical outlier detection and DINO structural filtering.
    Optionally visualizes the original pointcloud if original_pointcloud is set.
    """
    
    def __init__(self, files_info, file_key, clip_model_version="ViT-B/32", create_mesh=True, 
                 outlier_method='enhanced_adaptive', use_statistical_outliers=True, use_dino_filtering=True, original_pointcloud=None, config=None):
        if files_info is None or file_key not in files_info:
            raise ValueError("InteractiveTextSearch requires a valid featurized pointcloud. 'files_info' or 'file_key' is missing.")
        self.files_info = files_info
        self.file_key = file_key
        self.clip_model_version = clip_model_version
        self.create_mesh = create_mesh
        self.outlier_method = outlier_method
        self.use_statistical_outliers = use_statistical_outliers
        self.use_dino_filtering = use_dino_filtering
        self.original_pointcloud = original_pointcloud
        self.config = config
        
        # Load data once
        print("🔄 Loading pointcloud data...")
        # Support both .pt and numpy/ply variants
        file_entry = files_info[file_key]
        if isinstance(file_entry, str) and file_entry.endswith('.pt'):
            self.points, self.rgb, self.features_info = load_featurized_pointcloud(file_entry)
        else:
            self.points, self.rgb, self.features_info = load_featurized_pointcloud(files_info)

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
        # "vertical line at 1970-01-01" problem that happens when everything
        # is logged at the same instant with `static=True`.
        self.query_counter = 0
        
        # Pre-compute mesh for base pointcloud if requested
        self.base_mesh_vertices = None
        self.base_mesh_faces = None
        self.base_mesh_colors = None
        if self.create_mesh:
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
    
    def start_terminal_input(self):
        """Start terminal input thread."""
        def terminal_input_thread():
            print("\n" + "="*60)
            print("🎯 INTERACTIVE TEXT SEARCH READY")
            print("="*60)
            print("💡 How to search:")
            print(f"   • Type queries directly here and press Enter")
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
                        elif query.lower().startswith('grey_out_unmatched='):
                            grey_out_unmatched = query.split('=')[1].lower() == 'true'
                            self.query_queue.put(("grey_out_unmatched", grey_out_unmatched))
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
        print("📚 INTERACTIVE SEARCH HELP - ENHANCED ADAPTIVE METHOD")
        print("="*70)
        print("🔍 Search Commands:")
        print("  • <text>                    - Search for text (e.g., 'chair', 'red sofa')")
        print("  • clear                     - Clear all highlights")
        print("  • q                         - Quit")
        print("")
        print("🎯 Enhanced Adaptive Method (Default):")
        print("  • Combines CLIP semantic similarity with DINO structural coherence")
        print("  • Automatically adapts to object size and distribution")
        print("  • 🟠 Orange highlights show structured, coherent matches")
        print("  • Reduces scattered points, focuses on object boundaries")
        print("")
        print("🔗 DINO Structural Filtering:")
        if self.has_dino_features:
            print("  • use_dino_filtering=true   - Enable DINO structural filtering (default)")
            print("  • use_dino_filtering=false  - Use CLIP-only semantic search")
            print("    ↳ DINO helps identify coherent structures vs scattered points")
        else:
            print("  • DINO features not available - using CLIP-only mode")
        print("")
        print("📈 Alternative Outlier Detection Methods:")
        print("  • outlier_method=enhanced_adaptive - Enhanced adaptive with DINO (default)")
        print("  • outlier_method=adaptive   - Auto-select best method based on data")
        print("  • outlier_method=z_score    - Use Z-score method")
        print("  • outlier_method=percentile - Use percentile-based detection")
        print("  • outlier_method=combined   - Use combined methods")
        print("  • use_statistical_outliers=true/false - Toggle statistical mode")
        print("")
        print("🎨 Visualization:")
        print("  • Static RGB pointcloud: Always visible (base visualization)")
        print("  • 🟠 Orange highlights: Enhanced adaptive search results")
        print("  • Greyscale pointcloud: When grey_out_unmatched=true (focus mode)")
        print("  • show_clip_features=true: Enable CLIP feature visualization")
        print("  • show_all_methods_comparison=false: Disable all methods comparison")
        print("")
        print("🔬 All-Methods Comparison (Default):")
        print("  • Shows all outlier methods simultaneously with different colors")
        print("  • 🟢 Green: Percentile method (top 5%) | 🔵 Blue: Z-Score method (2 std devs)")
        print("  • 🟡 Yellow: Adaptive method (auto-selected) | 🟠 Orange: Enhanced Adaptive (DINO + structural coherence) | 🟣 Magenta: Combined method (union of all)")
        print("  • 🔴 Red: DINO-guided CLIP smoothing (CLIP-DINOiser)")
        print("  • Great for method comparison and analysis")
        print("  • Disable with: show_all_methods_comparison=false")
        print("")
        print("�� Traditional Method Settings:")
        print("  • threshold=0.25            - Set fixed similarity threshold")
        print("  • topk=100                  - Set maximum number of results")
        print("")
        print("💡 Method Comparison:")
        print("  • Enhanced Adaptive: Best results - semantic + structural coherence")
        print("    - CLIP finds semantic matches, DINO filters for structures")
        print("    - Reduces scattered points, focuses on object boundaries")
        print("  • Statistical CLIP-only: Good semantic matching with outlier detection")
        print("    - Adapts to object size automatically")
        print("  • Traditional: Fixed threshold, predictable but may need tuning")
        print("")
        print("🎮 Examples:")
        print("  • chair                     - Basic enhanced adaptive search")
        print("  • red wooden table          - Multi-word enhanced adaptive search")
        print("  • use_dino_filtering=false  - Switch to CLIP-only")
        print("  • outlier_method=z_score    - Switch outlier detection method")
        print("  • use_statistical_outliers=false - Switch to traditional")
        print("  • threshold=0.15            - Lower traditional threshold")
        print("  • grey_out_unmatched=true   - Enable focus mode (greyscale)")
        print("="*70 + "\n")

    def _log_readable_stats(self, query, num_results, threshold, method_used, stats=None):
        """
        Log a compact bullet-list summary of the latest search statistics to a dedicated
        `TextLog` entity so it appears as a tidy, readable panel in the Rerun viewer.
        """
        lines = [
            f"• Query: '{query}'" if query else "• Query: <none>",
            f"• Results: {num_results:,}",
            f"• Threshold: {threshold:.3f}",
            f"• Method: {method_used}",
        ]
        if stats:
            if 'mean' in stats:
                lines.append(f"• Mean sim: {stats['mean']:.3f}")
            if 'std' in stats:
                lines.append(f"• Std sim: {stats['std']:.3f}")
            if 'outlier_percentage' in stats:
                lines.append(f"• Outlier %: {stats['outlier_percentage']:.1f}")
        summary = "\n".join(lines)
        # Use `static=True` so this appears as a timeless text panel rather than a
        # densely stacked time-series.
        rr.log("stats/search/readable", rr.TextLog(summary, level=rr.TextLogLevel.INFO), static=True)

    def _log_stat_text(self, path: str, value):
        """Log a single numeric/statistic value as a timeless TextLog so it doesn't
        generate a time-series chart."""
        rr.log(path, rr.TextLog(str(value), level=rr.TextLogLevel.INFO), static=True)
    
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
            # Clear highlights - FOCUSED CLEANUP
            rr.log("world/text_similarity_highlights", rr.Clear(recursive=True))
            rr.log("world/text_similarity_highlights_rgb", rr.Clear(recursive=True))
            rr.log("world/clip_semantic_outliers", rr.Clear(recursive=True))
            rr.log("world/highlighted_mesh", rr.Clear(recursive=True))
            rr.log("world/all_methods_comparison", rr.Clear(recursive=True))
            self.last_highlight_indices = np.array([], dtype=int)  # Clear highlight indices
            # Clear search stats with better formatting
            rr.log("stats/search/current_query", rr.TextLog("No active search", level=rr.TextLogLevel.INFO))
            self._log_stat_text("stats/search/num_results", 0)
            self._log_stat_text("stats/search/top_similarity", 0.0)
            self._log_stat_text("stats/search/mean_similarity", 0.0)
            self._log_stat_text("stats/search/threshold", threshold)
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
                rr.set_time("timeline", timestamp=float(self.query_counter))

            # Note: for an empty query ("clear") we *don't* advance the counter –
            # this keeps the visualised stats aligned with the last executed
            # query.

            # FOCUSED APPROACH: Use only enhanced adaptive method (orange highlights)
            print(f"🎯 Running enhanced adaptive method for '{query}'...")
            
            # Run all methods comparison
            print(f"🎨 Running ALL outlier methods for comparison...")
            all_methods_results, similarities = run_all_outlier_methods(
                self.points,
                self.features_info['clip'],
                query,
                clip_encoder=self.clip_encoder,
                dino_features=self.features_info['dino'],
                min_threshold=0.05,
                percentile_threshold=95,
                iqr_multiplier=2.5,
                z_score_threshold=2.0,
                min_points=5
            )
            
            # Clear previous all-methods results
            rr.log("world/all_methods_comparison", rr.Clear(recursive=True))
            
            # Log each method's results with different colors
            method_summary = []
            total_points_found = 0
            for method_name, result in all_methods_results.items():
                # Format threshold safely for display
                threshold_display = f"{result['threshold']:.3f}" if result['threshold'] is not None else "N/A"
                if len(result['points']) > 0:
                    # Log points for this method
                    rr.log(f"world/all_methods_comparison/{method_name}", 
                            rr.Points3D(result['points'], colors=result['colors'], radii=0.008))

                    # --- Bounding box for dino_clip_smoothing ---
                    if method_name == 'dino_clip_smoothing' and len(result['points']) > 0:
                        pts = result['points']
                        bbox_min = pts.min(axis=0)
                        bbox_max = pts.max(axis=0)
                        # 8 corners of the box
                        corners = np.array([
                            [bbox_min[0], bbox_min[1], bbox_min[2]],
                            [bbox_max[0], bbox_min[1], bbox_min[2]],
                            [bbox_max[0], bbox_max[1], bbox_min[2]],
                            [bbox_min[0], bbox_max[1], bbox_min[2]],
                            [bbox_min[0], bbox_min[1], bbox_max[2]],
                            [bbox_max[0], bbox_min[1], bbox_max[2]],
                            [bbox_max[0], bbox_max[1], bbox_max[2]],
                            [bbox_min[0], bbox_max[1], bbox_max[2]],
                        ])
                        # Edges of the box (pairs of indices into corners)
                        edges = [
                            [0,1],[1,2],[2,3],[3,0], # bottom
                            [4,5],[5,6],[6,7],[7,4], # top
                            [0,4],[1,5],[2,6],[3,7]  # sides
                        ]
                        for i, (start, end) in enumerate(edges):
                            rr.log(f"world/all_methods_comparison/dino_clip_smoothing/bbox/edge_{i}", rr.LineStrips3D(np.array([corners[start], corners[end]]), colors=[[1,0,0]]))

                    # Add to summary
                    num_points = len(result['points'])
                    total_points_found += num_points
                    percentage = (num_points / len(self.points)) * 100
                    method_summary.append(f"• {method_name.upper()}: {num_points:,} points ({percentage:.1f}%) - threshold: {threshold_display}")
                    
                    # Log individual method stats
                    self._log_stat_text(f"stats/all_methods/{method_name}/num_points", num_points)
                    self._log_stat_text(f"stats/all_methods/{method_name}/percentage", round(percentage, 2))
                    self._log_stat_text(f"stats/all_methods/{method_name}/threshold", threshold_display)
                    if len(result['similarities']) > 0:
                        self._log_stat_text(f"stats/all_methods/{method_name}/max_similarity", round(float(result['similarities'].max()), 4))
                        self._log_stat_text(f"stats/all_methods/{method_name}/mean_similarity", round(float(result['similarities'].mean()), 4))
                else:
                    method_summary.append(f"• {method_name.upper()}: No outliers found")
                    self._log_stat_text(f"stats/all_methods/{method_name}/num_points", 0)
                    self._log_stat_text(f"stats/all_methods/{method_name}/percentage", 0.0)
                    self._log_stat_text(f"stats/all_methods/{method_name}/threshold", threshold_display)
            
            # Log summary text with legend
            legend_text = """🎨 COLOR LEGEND:
🔴 RED: IQR method (Interquartile Range)
🟢 GREEN: Percentile method (top 5%)
🔵 BLUE: Z-Score method (2 std devs)
🟡 YELLOW: Adaptive method (auto-selected)
🟠 ORANGE: Enhanced Adaptive (DINO + structural coherence)
🟣 MAGENTA: Combined method (union of all)
🟥 BRIGHT RED: DINO-guided CLIP smoothing (CLIP-DINOiser)"""
            
            summary_text = f"""🔬 ALL OUTLIER METHODS COMPARISON: '{query}'
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📊 Total points found across all methods: {total_points_found:,} / {len(self.points):,} points

{legend_text}

📈 METHOD RESULTS:
{chr(10).join(method_summary)}

💡 TIP: Toggle visibility in Rerun to compare methods side-by-side
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"""
            rr.log("docs/all_methods_summary", rr.TextLog(summary_text, level=rr.TextLogLevel.INFO))
            # Log overall comparison stats
            self._log_stat_text("stats/all_methods/total_points_found", total_points_found)
            self._log_stat_text("stats/all_methods/total_percentage", round((total_points_found / len(self.points)) * 100, 2))
            # Store results for potential use
            self.all_methods_results = all_methods_results
            self.last_similarities = similarities
            # Console output for all-methods comparison
            print(f"🎨 All outlier methods comparison completed for '{query}'")
            print(f"   📊 Total points found: {total_points_found:,} across all methods")
            print(f"   🎯 Check Rerun viewer for color-coded results:")
            print(f"      🔴 Red: IQR method")
            print(f"      🟢 Green: Percentile method") 
            print(f"      🔵 Blue: Z-Score method")
            print(f"      🟡 Yellow: Adaptive method")
            print(f"      🟠 Orange: Enhanced Adaptive (DINO + structural)")
            print(f"      🟣 Magenta: Combined method")
            print(f"      🟥 Bright Red: DINO-guided CLIP smoothing (CLIP-DINOiser)")
            print(f"   💡 Toggle visibility in Rerun to compare methods side-by-side")

            # --- CONSOLIDATED HIGHLIGHT LOGIC ---
            # Always use enhanced_adaptive for main highlights
            enhanced = all_methods_results.get('enhanced_adaptive', None)
            if enhanced and len(enhanced['points']) > 0:
                highlight_indices = enhanced['indices']
                highlight_points = enhanced['points']
                highlight_colors = enhanced['colors']
                similarities = enhanced['similarities']
                outlier_stats = enhanced['stats']
                method_used = enhanced.get('method_used', 'enhanced_adaptive')
                display_threshold = outlier_stats.get('threshold_used', 0.0) if outlier_stats else 0.0
                self.last_similarities = similarities
                self.last_outlier_stats = outlier_stats
                self.last_highlight_indices = highlight_indices
                # Log the enhanced adaptive highlights (orange)
                rr.log("world/text_similarity_highlights", 
                       rr.Points3D(highlight_points, colors=highlight_colors, radii=0.01))
                # Clear redundant visualizations
                rr.log("world/text_similarity_highlights_rgb", rr.Clear(recursive=True))
                rr.log("world/clip_semantic_outliers", rr.Clear(recursive=True))
                # Create mesh for highlighted points if mesh creation is enabled (OPTIONAL)
                if self.create_mesh and len(highlight_points) > 100:
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
                self._log_stat_text("stats/search/num_results", len(highlight_points))
                self._log_stat_text("stats/search/top_similarity", round(float(similarities.max()), 4))
                self._log_stat_text("stats/search/mean_similarity", round(float(similarities.mean()), 4))
                self._log_stat_text("stats/search/threshold", round(display_threshold, 4))
                rr.log("stats/search/detection_method", rr.TextLog(method_used))
                # Optionally, log more stats as before...
                self.log_adaptive_pointcloud()
            else:
                # No results found
                print(f"❌ No matches found for '{query}' (enhanced_adaptive)")
                rr.log("stats/search/current_query", rr.TextLog(f"Query: '{query}' (no results)", level=rr.TextLogLevel.WARN))
                self._log_stat_text("stats/search/num_results", 0)
                self._log_stat_text("stats/search/top_similarity", 0.0)
                self._log_stat_text("stats/search/mean_similarity", 0.0)
                self._log_stat_text("stats/search/threshold", 0.0)
                rr.log("stats/search/detection_method", rr.TextLog("enhanced_adaptive"))
                # Clear all highlights
                rr.log("world/text_similarity_highlights", rr.Clear(recursive=True))
                rr.log("world/text_similarity_highlights_rgb", rr.Clear(recursive=True))
                rr.log("world/clip_semantic_outliers", rr.Clear(recursive=True))
                rr.log("world/highlighted_mesh", rr.Clear(recursive=True))
                self.last_highlight_indices = np.array([], dtype=int)
                self.log_adaptive_pointcloud()
        except Exception as e:
            print(f"❌ Error processing query '{query}': {e}")
            rr.log("errors/search", rr.TextLog(f"Error: {str(e)}", level=rr.TextLogLevel.ERROR))
        finally:
            # At the end of the method, always update the adaptive pointcloud
            self.log_adaptive_pointcloud()
    
    def run_interactive_session(self, mode="local", port=9878, scripted_queries=None):
        """
        Run the main session. 
        If scripted_queries is provided, it runs them in sequence.
        Otherwise, it starts an interactive terminal session.
        """
        # Initialize Rerun
        rr.init("Interactive_Text_Search", spawn=False)

        if mode == "remote" and not getattr(self.config, 'rerun_save_enabled', False):
            uri = rr.serve_grpc(grpc_port=port)
            print(f"🌐 Rerun grpc server started on {uri}")
        elif mode == "local" and not getattr(self.config, 'rerun_save_enabled', False):
            rr.spawn(port=port)
            print(f"🌐 Rerun viewer spawned on port {port}")
        elif getattr(self.config, 'rerun_save_enabled', False):
            print(f"💾 Rerun recording will be saved to {getattr(self.config, 'rerun_save_path', 'output.rrd')}")
        else:
            raise ValueError(f"Invalid rerun servermode: {mode}")
        
        # Setup coordinate frame
        rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Y_UP, static=True)
        rr.set_time("timeline", timestamp=0.0)

        # Visualize original pointcloud if requested
        if self.original_pointcloud is not None:
            print("\n📂 Visualizing original pointcloud (PLY) for reference...")
            try:
                points, colors = load_original_pointcloud(self.file_key, self.original_pointcloud)
                if colors is not None:
                    rr.log("world/original_pointcloud", rr.Points3D(points, colors=colors, radii=0.005), static=True)
                else:
                    rr.log("world/original_pointcloud", rr.Points3D(points, radii=0.005), static=True)
                print("Original pointcloud logged to rerun.")
            except Exception as e:
                print(f"[Warning] Could not visualize original pointcloud: {e}")
        else:
            print("[Info] No original_pointcloud path provided; skipping original pointcloud visualization.")
        
        # Log the static RGB pointcloud once (never changes) - ALWAYS SHOWN
        rr.log("world/pointcloud_rgb_static", rr.Points3D(self.points, colors=self.rgb, radii=0.005), static=True)

        # Log the adaptive pointcloud (updates after each query) - CONDITIONAL
        def log_adaptive_pointcloud():
            if bool(self.config.interactive_search.grey_out_unmatched) and hasattr(self, 'last_highlight_indices') and len(self.last_highlight_indices) > 0:
                highlight_indices = self.last_highlight_indices
                # Convert all points to greyscale
                rgb = self.rgb
                luminance = 0.299 * rgb[:, 0] + 0.587 * rgb[:, 1] + 0.114 * rgb[:, 2]
                greyscale_colors = np.stack([luminance, luminance, luminance], axis=1)
                # Restore original RGB for highlighted points
                greyscale_colors[highlight_indices] = rgb[highlight_indices]
                rr.log("world/pointcloud_grey", rr.Points3D(self.points, colors=greyscale_colors, radii=0.005))
            else:
                # Clear greyscale when not needed
                rr.log("world/pointcloud_grey", rr.Clear(recursive=True))
        self.log_adaptive_pointcloud = log_adaptive_pointcloud
        self.log_adaptive_pointcloud()

        # Log base mesh if available (OPTIONAL - keep as user doesn't mind)
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
        
        # Log CLIP feature visualization (OPTIONAL - make configurable)
        print("🎨 Logging CLIP feature visualization...")
        clip_colors = features_to_colors_pca(self.features_info['clip'], method='hsv')
        rr.log("world/pointcloud_clip_features", 
                rr.Points3D(self.points, colors=clip_colors, radii=0.005), 
                   static=True)
        # Log DINO feature visualization (OPTIONAL)
        print("🦕 Logging DINO feature visualization...")
        dino_colors = features_to_colors_pca(self.features_info['dino'], method='hsv')
        rr.log("world/pointcloud_dino_features", 
                rr.Points3D(self.points, colors=dino_colors, radii=0.005), 
                static=True)
        
        # Log comprehensive initial stats with better organization
        bbox_min = self.points.min(axis=0)
        bbox_max = self.points.max(axis=0)
        bbox_size = bbox_max - bbox_min
        
        # Log the same numerical stats as timeless TextLogs for readability.
        self._log_stat_text("stats/pointcloud/total_points", len(self.points))
        self._log_stat_text("stats/pointcloud/clip_feature_dim", self.features_info['clip'].shape[1])
        self._log_stat_text("stats/pointcloud/bbox_size_x", round(float(bbox_size[0]), 4))
        self._log_stat_text("stats/pointcloud/bbox_size_y", round(float(bbox_size[1]), 4))
        self._log_stat_text("stats/pointcloud/bbox_size_z", round(float(bbox_size[2]), 4))
        self._log_stat_text("stats/pointcloud/bbox_volume", round(float(np.prod(bbox_size)), 4))
        
        if self.base_mesh_vertices is not None and self.base_mesh_faces is not None:
            self._log_stat_text("stats/pointcloud/mesh_vertices", len(self.base_mesh_vertices))
            self._log_stat_text("stats/pointcloud/mesh_faces", len(self.base_mesh_faces))
        
        # Create a comprehensive pointcloud summary
        mesh_info = ""
        if self.base_mesh_vertices is not None and self.base_mesh_faces is not None:
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
   • Adjust settings: "threshold=0.15", "topk=300"
   • Type "help" for more commands

🎨 Visualization Colors:
   • Blue points: CLIP semantic matches (raw semantic similarity)
   • Gold points: DINO filtered results (structured, coherent matches)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"""
        
        rr.log("docs/instructions", rr.TextLog(pointcloud_summary, level=rr.TextLogLevel.INFO), static=True)
        
        # Initialize search stats structure
        rr.log("stats/search/current_query", rr.TextLog("No active search", level=rr.TextLogLevel.INFO), static=True)
        # Readable search defaults
        self._log_stat_text("stats/search/num_results", 0)
        self._log_stat_text("stats/search/threshold", 0.2)
        self._log_stat_text("stats/search/top_k", 200)
        
        # Settings
        current_threshold = 0.2
        current_top_k = 200
        current_outlier_method = self.outlier_method
        current_use_statistical_outliers = self.use_statistical_outliers
        current_use_dino_filtering = self.use_dino_filtering

        # --- SCRIPTED MODE ---
        if scripted_queries:
            print("🚀 Running in scripted mode...")
            for i, query in enumerate(scripted_queries):
                print(f"\n--- Processing query {i+1}/{len(scripted_queries)}: '{query}' ---")
                self.process_text_query(
                    query, 
                    current_top_k, 
                    current_threshold, 
                    current_outlier_method, 
                    current_use_statistical_outliers, 
                    current_use_dino_filtering
                )
                time.sleep(2) # Pause for 2 seconds to make the timeline playback visually clear

            print("\n✅ Scripted run complete. The Rerun viewer is now static.")
            # Save the Rerun recording if enabled in config
            if self.config is not None and getattr(self.config, 'rerun_save_enabled', False):
                save_path = getattr(self.config, 'rerun_save_path', 'output.rrd')
                rr.save(save_path)
                print(f"💾 Saving Rerun recording to {save_path} ...")
                sys.exit()
            # Keep the script alive so you can explore the viewer
            print("Press Ctrl+C to exit.")
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                print("\n⏹️ Session ended.")

        # --- INTERACTIVE MODE ---
        else:
            terminal_thread = self.start_terminal_input()
            try:
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
                            current_use_statistical_outliers = str(query).lower() == 'true'
                            print(f"🎛️  Statistical outlier detection: {current_use_statistical_outliers}")
                            # Re-process current query with new outlier method
                            if self.current_query:
                                self.process_text_query(self.current_query, current_top_k, current_threshold, current_outlier_method, current_use_statistical_outliers, current_use_dino_filtering)
                        
                        elif source == "use_dino_filtering":
                            current_use_dino_filtering = str(query).lower() == 'true'
                            print(f"🎛️  DINO structural filtering: {current_use_dino_filtering}")
                            # Re-process current query with new DINO filtering setting
                            if self.current_query:
                                self.process_text_query(self.current_query, current_top_k, current_threshold, current_outlier_method, current_use_statistical_outliers, current_use_dino_filtering)
                        
                        elif source == "grey_out_unmatched":
                            # Always store as boolean
                            self.config.interactive_search.grey_out_unmatched = bool(query)
                            print(f"🎛️  Grey out unmatched: {self.config.interactive_search.grey_out_unmatched}")
                            # Re-log the adaptive pointcloud with the new setting
                            self.log_adaptive_pointcloud()
                            if self.current_query:
                                self.process_text_query(self.current_query, current_top_k, current_threshold, current_outlier_method, current_use_statistical_outliers, current_use_dino_filtering)
                        
                        else:
                            # Regular text query
                            self.current_query = query
                            self.process_text_query(query, current_top_k, current_threshold, current_outlier_method, current_use_statistical_outliers, current_use_dino_filtering)
                            
                    except queue.Empty:
                        continue
                    except KeyboardInterrupt:
                        break
                        
            except KeyboardInterrupt:
                print("\n⏹️  Interactive session interrupted")
            finally:
                # Save the Rerun recording if enabled in config
                if self.config is not None and getattr(self.config, 'rerun_save_enabled', False):
                    save_path = getattr(self.config, 'rerun_save_path', 'output.rrd')
                    print(f"💾 Saving Rerun recording to {save_path} ...")
                    rr.save(save_path)
                self.running = False
                print("🛑 Interactive session ended") 

def get_adaptive_weights_by_scatter(points, initial_outliers, default_clip=0.6, default_struct=0.4):
    selected_points = points[initial_outliers]
    spatial_var = np.var(selected_points, axis=0).mean()
    # Example thresholds (tune as needed)
    if spatial_var > 0.5:
        return 0.5, 0.5
    elif spatial_var < 0.1:
        return 0.7, 0.3
    else:
        return default_clip, default_struct

def batched_knn(x, k, batch_size=500):
    """
    Compute kNN indices for x in batches to avoid OOM.
    Args:
        x: (N, D) torch tensor (should be normalized if using cosine/L2)
        k: int, number of neighbors
        batch_size: int, batch size for processing
    Returns:
        knn_indices: (N, k) torch tensor of neighbor indices
    """
    N = x.shape[0]
    device = x.device
    knn_indices = []
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        batch = x[start:end]  # (B, D)
        dists = torch.cdist(batch, x)  # (B, N)
        topk = dists.topk(k, largest=False)
        knn_indices.append(topk.indices.cpu())
    return torch.cat(knn_indices, dim=0)

def dino_guided_clip_smoothing(points, clip_features, dino_features, text_features, threshold=0.2, k=10, mix_alpha=0.5, N_top=5000, batch_size=200, cluster_eps=0.05, cluster_min_samples=10):
    import numpy as np
    from sklearn.cluster import DBSCAN
    N = points.shape[0]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Convert to torch
    clip_feats = torch.from_numpy(clip_features).float().to(device)
    dino_feats = torch.from_numpy(dino_features).float().to(device)
    text_emb = torch.from_numpy(text_features).float().to(device).view(-1)
    # Normalize
    clip_feats = F.normalize(clip_feats, dim=-1)
    text_emb = F.normalize(text_emb, dim=-1)
    dino_norm = F.normalize(dino_feats, dim=-1)
    # CLIP similarity
    clip_sim = (clip_feats @ text_emb).view(-1)  # (N,)
    # kNN in DINO space (batched) for top-N points
    N_top = min(N_top, N)
    top_indices = torch.topk(clip_sim, N_top).indices
    dino_subset = dino_norm[top_indices]
    clip_subset = clip_sim[top_indices]
    knn_indices = batched_knn(dino_subset, k, batch_size=batch_size).to(device)  # (N_top, k)
    # DINO affinity (cosine) for subset
    affinity = torch.einsum('nd,nkd->nk', dino_subset, dino_subset[knn_indices])  # (N_top, k)
    affinity = (affinity + 1) / 2  # [0,1]
    # Smoothing for subset
    neighbor_sims = clip_subset[knn_indices]  # (N_top, k)
    weighted_sims = affinity * neighbor_sims  # (N_top, k)
    smoothed_sim = weighted_sims.sum(dim=1) / (affinity.sum(dim=1) + 1e-8)  # (N_top,)
    smoothed_sim = smoothed_sim.view(-1, 1)
    # Optionally mix with original
    final_sim = mix_alpha * clip_subset.view(-1, 1) + (1 - mix_alpha) * smoothed_sim  # (N_top, 1)
    final_sim_np = final_sim.cpu().numpy().flatten()
    mask = (final_sim_np > threshold)
    selected_indices = top_indices.cpu().numpy()[mask]  # indices into full point cloud
    highlight_points = points[selected_indices]
    highlight_similarities = final_sim_np[mask]
    # --- Clustering: extract largest spatial cluster from highlight_points ---
    if len(highlight_points) > 0:
        clustering = DBSCAN(eps=cluster_eps, min_samples=cluster_min_samples).fit(highlight_points)
        labels = clustering.labels_
        valid = labels != -1
        if np.any(valid):
            unique, counts = np.unique(labels[valid], return_counts=True)
            largest = unique[np.argmax(counts)]
            cluster_indices = np.where(labels == largest)[0]  # indices into highlight_points
            final_indices = selected_indices[cluster_indices]  # indices into full point cloud
            final_points = points[final_indices]
            final_similarities = highlight_similarities[cluster_indices]
        else:
            final_indices = np.array([], dtype=int)
            final_points = np.array([], dtype=points.dtype).reshape(0, 3)
            final_similarities = np.array([])
    else:
        final_indices = np.array([], dtype=int)
        final_points = np.array([], dtype=points.dtype).reshape(0, 3)
        final_similarities = np.array([])
    # Color: red, intensity by similarity
    num_highlights = len(final_indices)
    highlight_colors = np.zeros((num_highlights, 3))
    if num_highlights > 0:
        min_sim = final_similarities.min()
        max_sim = final_similarities.max()
        for i, sim in enumerate(final_similarities):
            if max_sim > min_sim:
                intensity = 0.4 + 0.6 * (sim - min_sim) / (max_sim - min_sim)
            else:
                intensity = 1.0
            highlight_colors[i] = [intensity, 0.0, 0.0]  # Red
    stats = {
        'threshold': threshold,
        'num_points': num_highlights,
        'k': k,
        'mix_alpha': mix_alpha,
        'mean_sim': float(final_sim_np.mean()),
        'max_sim': float(final_sim_np.max()),
        'min_sim': float(final_sim_np.min()),
        'clustering': {
            'eps': cluster_eps,
            'min_samples': cluster_min_samples,
            'num_clusters': len(np.unique(labels[labels != -1])) if len(highlight_points) > 0 and np.any(valid) else 0,
            'largest_cluster_size': len(cluster_indices) if len(highlight_points) > 0 and np.any(valid) else 0
        }
    }
    print(f"Using device: {device}")
    print(f"Smoothed sim stats: min={final_sim_np.min()}, max={final_sim_np.max()}, mean={final_sim_np.mean()}")
    return final_indices, final_points, highlight_colors, final_sim_np, threshold, 'dino_clip_smoothing', stats