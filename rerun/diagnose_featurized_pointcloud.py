import torch
import rerun as rr
from sklearn.decomposition import PCA
import clip
import torch.nn.functional as F
import argparse
import open3d as o3d
import os
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
import numpy as np


# ---------- Helpers ----------

def log_pointcloud(name, points, colors, radii=0.01):
    rr.log(name, rr.Points3D(positions=points, colors=colors, radii=radii))

def pca_to_rgb(features):
    pca = PCA(n_components=3).fit_transform(features)
    pca = (pca - pca.min(axis=0)) / (pca.max(axis=0) - pca.min(axis=0) + 1e-6)
    print(f"PCA stats → min: {pca.min(axis=0)}, max: {pca.max(axis=0)}")

    return pca

def select_params_from_mask_size(points, idx_adaptive):
    """
    Adapt DBSCAN/KNN params based on the spread and size of the percentile mask points.
    """
    if len(idx_adaptive) == 0:
        # Fallback
        return 0.3, 10, 3

    # Spatial spread heuristic
    spread = np.linalg.norm(points[idx_adaptive].max(axis=0) - points[idx_adaptive].min(axis=0))

    if len(idx_adaptive) > 3000 or spread > 1.0:
        # Likely large object
        dbscan_eps = 0.6
        knn_k = 30
        min_samples = 10
    elif len(idx_adaptive) > 1000 or spread > 0.5:
        # Medium object
        dbscan_eps = 0.4
        knn_k = 20
        min_samples = 5
    else:
        # Small object
        dbscan_eps = 0.3
        knn_k = 10
        min_samples = 3

    return dbscan_eps, knn_k, min_samples


def compute_clip_similarity(features_clip, text_query, model, device):
    with torch.no_grad():
        text_feat = model.encode_text(clip.tokenize([text_query]).to(device))
        text_feat = F.normalize(text_feat, dim=-1)

        features_clip = (
            features_clip if isinstance(features_clip, torch.Tensor)
            else torch.tensor(features_clip)
        ).to(device)
        features_clip = F.normalize(features_clip, dim=-1)

        # Ensure both are the same dtype
        features_clip = features_clip.to(text_feat.dtype)

        similarity = (features_clip @ text_feat.T).squeeze().cpu().numpy()
        return similarity

def refine_with_zscore_and_knn(points, zscore_indices, percentile_indices, 
                               knn_k=20, dbscan_eps=0.3, dbscan_min_samples=5):
    """
    Refines a noisy percentile CLIP mask using high-confidence z-score anchors.
    Args:
        points: [N, 3] numpy array of point positions.
        zscore_indices: Indices of high-confidence points (anchors).
        percentile_indices: Indices of broader noisy points (percentile mask).
        knn_k: Number of neighbors for local expansion.
        dbscan_eps: DBSCAN eps for cleanup.
        dbscan_min_samples: DBSCAN min_samples for cleanup.
    Returns:
        final_indices: Cleaned indices of points likely part of the object.
    """
    if len(zscore_indices) == 0 or len(percentile_indices) == 0:
        return zscore_indices  # Nothing to refine

    # Build KNN from percentile points
    knn = NearestNeighbors(n_neighbors=knn_k)
    knn.fit(points[percentile_indices])

    # For each z-score point, find KNN neighbors within percentile points
    _, neighbor_indices = knn.kneighbors(points[zscore_indices])
    expanded_indices = percentile_indices[np.unique(neighbor_indices.flatten())]

    # Run DBSCAN on expanded points for spatial cleanup
    expanded_points = points[expanded_indices]
    clustering = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples).fit(expanded_points)
    labels = clustering.labels_

    # Keep largest cluster
    unique_labels, counts = np.unique(labels[labels >= 0], return_counts=True)
    if len(unique_labels) == 0:
        return expanded_indices  # Fallback: no clusters, return all expanded

    largest_cluster_label = unique_labels[np.argmax(counts)]
    final_indices = expanded_indices[labels == largest_cluster_label]

    return final_indices
    
def filter_with_dino_dbscan(points, dino_features, indices, eps=0.1, min_samples=10, spatial_weight=0.3):
    """
    Filter selected points using DBSCAN over combined DINO + spatial features.

    Returns:
        filtered_indices: subset of `indices` that are part of coherent clusters
    """
    if len(indices) < min_samples:
        return indices  # too few to cluster

    selected_points = points[indices]
    selected_dino = dino_features[indices]

    # Normalize features
    dino_norm = selected_dino / (np.linalg.norm(selected_dino, axis=1, keepdims=True) + 1e-8)

    # Normalize spatial coords
    spatial = selected_points
    spatial_range = spatial.max(axis=0) - spatial.min(axis=0)
    spatial_range[spatial_range == 0] = 1
    spatial_norm = (spatial - spatial.min(axis=0)) / spatial_range

    # Combine with spatial weighting
    fused = np.concatenate([dino_norm, spatial_norm * spatial_weight], axis=1)

    # Run DBSCAN
    cluster_labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(fused)

    # Keep points in clusters (not noise)
    kept = cluster_labels >= 0
    return indices[kept]



# ---------- Main ----------

def main(args):
    rr.init("3D Search", spawn=True, default_port=9878)

    # Optional: log original .ply geometry
    if args.ply_path:
        print(f":card_index_dividers:  Logging original PLY: {args.ply_path}")
        pcd = o3d.io.read_point_cloud(args.ply_path)
        points_ply = np.asarray(pcd.points)
        colors_ply = np.asarray(pcd.colors) if pcd.has_colors() else np.ones_like(points_ply) * 0.8
        log_pointcloud("world/original_pointcloud", points_ply, colors_ply, radii=0.007)

    # Load featurized pointcloud
    print(f":package: Loading: {args.pt_path}")
    data = torch.load(args.pt_path, map_location='cpu')
    points = data["points"].numpy()
    rgb = data["rgb"].numpy()
    if rgb.max() > 1.0: rgb = rgb / 255.0
    log_pointcloud("world/voxelized_pointcloud", points, rgb, radii=0.02)

    # Visualize any available features (CLIP, DINO, SAM)
    for key in ["features_clip", "features_dino", "features_sam"]:
        if key in data:
            print(f":art: Visualizing {key} as PCA")
            features = data[key].cpu().numpy().astype(np.float32)
            colors = pca_to_rgb(features)
            log_pointcloud(f"world/{key}_pca", points, colors)

    # Optional: run CLIP similarity search
    if "features_clip" in data:
        print(f":mag: Matching CLIP features to: '{args.query}'")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        clip_dim = data["features_clip"].shape[1]
        clip_model_name = "ViT-L/14" if clip_dim == 768 else "ViT-B/32"
        print(f"Auto-detected CLIP model: {clip_model_name} for {clip_dim}D features")
        model, _ = clip.load(clip_model_name, device=device)

        features_clip = data["features_clip"]
        if not isinstance(features_clip, torch.Tensor):
            features_clip = torch.tensor(features_clip)

        def run_query(text_query):
            paths_to_clear = [
                "world/clip_percentile",
                "world/clip_zscore",
                "world/clip_hybrid_raw",
                "world/clip_hybrid_refined",
                "world/refined_query",
                "world/clip_topk",
            ]

            for path in paths_to_clear:
                rr.log(path, rr.Clear(recursive=True))

            sim = compute_clip_similarity(features_clip, text_query, model, device)
            
            # --- Top-k ---
            topk_idx = sim.argsort()[-args.topk:]
            log_pointcloud("world/clip_topk (red)", points[topk_idx],
                        np.tile([[1.0, 0.2, 0.2]], (len(topk_idx), 1)), radii=0.025)

            # --- Adaptive (Percentile) ---
            perc_thresh = np.percentile(sim, 90)
            idx_adaptive = np.where(sim >= perc_thresh)[0]
            if len(idx_adaptive) < 20:
                idx_adaptive = sim.argsort()[-20:]
                perc_thresh = sim[idx_adaptive[0]]
            log_pointcloud("world/clip_percentile (yellow)", points[idx_adaptive],
                        np.tile([[1.0, 0.8, 0.1]], (len(idx_adaptive), 1)), radii=0.025)

            dbscan_eps, knn_k, min_samples = select_params_from_mask_size(points, idx_adaptive)

            # --- Z-Score ---
            mean = sim.mean()
            std = sim.std()
            z_thresh = mean + 2 * std
            idx_zscore = np.where(sim > z_thresh)[0]
            if len(idx_zscore) == 0:
                idx_zscore = sim.argsort()[-20:]
                z_thresh = sim[idx_zscore[0]]
            log_pointcloud("world/clip_zscore (blue)", points[idx_zscore],
                        np.tile([[0.2, 0.5, 1.0]], (len(idx_zscore), 1)), radii=0.025)

            refined_indices = refine_with_zscore_and_knn(
                points,
                idx_zscore,
                idx_adaptive,
                knn_k=knn_k,
                dbscan_eps=dbscan_eps,
                dbscan_min_samples=min_samples
            )
            log_pointcloud("world/refined_query (red)", points[refined_indices],
                np.tile([[1.0, 0.0, 0.0]], (len(refined_indices), 1)), radii=0.03)


            # --- Hybrid (fixed min threshold + fallback) ---
            hybrid_thresh = 0.25
            idx_hybrid = np.where(sim >= hybrid_thresh)[0]
            if len(idx_hybrid) < 20:
                idx_hybrid = sim.argsort()[-20:]
                hybrid_thresh = sim[idx_hybrid[0]]
            log_pointcloud("world/clip_hybrid_raw (turquoise)", points[idx_hybrid],
    np.tile([[0.0, 1.0, 1.0]], (len(idx_hybrid), 1)), radii=0.025)

            # Filtered to pick biggest cluster
            hybrid_points = points[idx_hybrid]
            dbscan = DBSCAN(eps=0.3, min_samples=10)
            labels = dbscan.fit_predict(hybrid_points)
            unique_labels, counts = np.unique(labels[labels >= 0], return_counts=True)
            if len(unique_labels) > 0:
                largest_cluster_label = unique_labels[np.argmax(counts)]
                final_indices = idx_hybrid[labels == largest_cluster_label]
            else:
                final_indices = idx_hybrid
            log_pointcloud("world/clip_hybrid_refined (green)", points[final_indices],
               np.tile([[0.2, 1.0, 0.4]], (len(final_indices), 1)), radii=0.025)

            dino_features = data["features_dino"].cpu().numpy()
            refined_indices = filter_with_dino_dbscan(
                points, dino_features, refined_indices,
                eps=0.1, min_samples=10, spatial_weight=0.3
            )
            log_pointcloud("world/refined_query_dino (purple)", points[refined_indices],
                np.tile([[0.8, 0.2, 1.0]], (len(refined_indices), 1)), radii=0.03)

            
        # Run CLI query if given, then continue into interactive loop
        if args.query:
            print(f":mag: One-shot query: '{args.query}'")
            run_query(args.query)

        print("\n:speech_balloon: Enter additional queries (or 'q' to quit):")
        while True:
            try:
                user_input = input(":mag: Query: ").strip()
                if user_input.lower() == 'q':
                    break
                if user_input:
                    run_query(user_input)
            except KeyboardInterrupt:
                break



# ---------- CLI ----------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("pt_path", help=".pt file with featurized point cloud")
    parser.add_argument("--query", type=str, help="Text query for CLIP similarity")
    parser.add_argument("--topk", type=int, default=1000, help="Top-k matches to highlight")
    parser.add_argument("--ply-path", type=str, help="Optional original .ply pointcloud path")
    args = parser.parse_args()
    main(args)