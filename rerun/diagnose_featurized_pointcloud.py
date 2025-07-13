import torch
import rerun as rr
import numpy as np
from sklearn.decomposition import PCA
import clip
import torch.nn.functional as F
import argparse
import open3d as o3d
import os
from sklearn.cluster import DBSCAN


# ---------- Helpers ----------

def log_pointcloud(name, points, colors, radii=0.01):
    rr.log(name, rr.Points3D(positions=points, colors=colors, radii=radii))

def pca_to_rgb(features):
    pca = PCA(n_components=3).fit_transform(features)
    pca = (pca - pca.min(axis=0)) / (pca.max(axis=0) - pca.min(axis=0) + 1e-6)
    print(f"PCA stats → min: {pca.min(axis=0)}, max: {pca.max(axis=0)}")

    return pca

def compute_clip_similarity(features_clip, text_query, model, device):
    with torch.no_grad():
        text_feat = model.encode_text(clip.tokenize([text_query]).to(device))
        text_feat = F.normalize(text_feat, dim=-1)

        features_clip = (
            features_clip if isinstance(features_clip, torch.Tensor)
            else torch.tensor(features_clip)
        ).to(device)
        features_clip = F.normalize(features_clip, dim=-1)

        similarity = (features_clip @ text_feat.T).squeeze().cpu().numpy()
        return similarity
    
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
    rr.init("3D Search", spawn=True)

    # Optional: log original .ply geometry
    if args.ply_path:
        print(f"🗂️  Logging original PLY: {args.ply_path}")
        pcd = o3d.io.read_point_cloud(args.ply_path)
        points_ply = np.asarray(pcd.points)
        colors_ply = np.asarray(pcd.colors) if pcd.has_colors() else np.ones_like(points_ply) * 0.8
        log_pointcloud("world/original_pointcloud", points_ply, colors_ply, radii=0.007)

    # Load featurized pointcloud
    print(f"📦 Loading: {args.pt_path}")
    data = torch.load(args.pt_path, map_location='cpu')
    points = data["points"].numpy()
    rgb = data["rgb"].numpy()
    if rgb.max() > 1.0: rgb = rgb / 255.0
    log_pointcloud("world/voxelized_pointcloud", points, rgb, radii=0.05)

    # Visualize any available features (CLIP, DINO, SAM)
    for key in ["features_clip", "features_dino", "features_sam"]:
        if key in data:
            print(f"🎨 Visualizing {key} as PCA")
            features = data[key].cpu().numpy().astype(np.float32)
            colors = pca_to_rgb(features)
            log_pointcloud(f"world/{key}_pca", points, colors)

    # Optional: run CLIP similarity search
    if "features_clip" in data:
        print(f"🔍 Matching CLIP features to: '{args.query}'")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        clip_dim = data["features_clip"].shape[1]
        clip_model_name = "ViT-L/14" if clip_dim == 768 else "ViT-B/32"
        print(f"Auto-detected CLIP model: {clip_model_name} for {clip_dim}D features")
        model, _ = clip.load(clip_model_name, device=device)

        features_clip = data["features_clip"]
        if not isinstance(features_clip, torch.Tensor):
            features_clip = torch.tensor(features_clip)

        def run_query(text_query):
            rr.log("query/current", rr.TextDocument(f"🔎 Current query: '{text_query}'"))
            rr.log("world/clip_text_topk", rr.Clear(recursive=True))
            rr.log("world/clip_text_adaptive", rr.Clear(recursive=True))
            rr.log("world/clip_text_zscore", rr.Clear(recursive=True))
            rr.log("world/clip_text_hybrid", rr.Clear(recursive=True))

            sim = compute_clip_similarity(features_clip, text_query, model, device)

            # --- Top-k ---
            topk_idx = sim.argsort()[-args.topk:]
            log_pointcloud("world/clip_text_topk (red)", points[topk_idx],
                        np.tile([[1.0, 0.2, 0.2]], (len(topk_idx), 1)), radii=0.025)

            # --- Adaptive (Percentile) ---
            perc_thresh = np.percentile(sim, 99)
            idx_adaptive = np.where(sim >= perc_thresh)[0]
            if len(idx_adaptive) < 20:
                idx_adaptive = sim.argsort()[-20:]
                perc_thresh = sim[idx_adaptive[0]]
            log_pointcloud("world/clip_text_adaptive (yellow)", points[idx_adaptive],
                        np.tile([[1.0, 0.8, 0.1]], (len(idx_adaptive), 1)), radii=0.025)

            # --- Z-Score ---
            mean = sim.mean()
            std = sim.std()
            z_thresh = mean + 2.5 * std
            idx_zscore = np.where(sim > z_thresh)[0]
            if len(idx_zscore) == 0:
                idx_zscore = sim.argsort()[-20:]
                z_thresh = sim[idx_zscore[0]]
            log_pointcloud("world/clip_text_zscore (blue)", points[idx_zscore],
                        np.tile([[0.2, 0.5, 1.0]], (len(idx_zscore), 1)), radii=0.025)

            # --- Hybrid (fixed min threshold + fallback) ---
            hybrid_thresh = 0.25
            idx_hybrid = np.where(sim >= hybrid_thresh)[0]
            if len(idx_hybrid) < 20:
                idx_hybrid = sim.argsort()[-20:]
                hybrid_thresh = sim[idx_hybrid[0]]
            log_pointcloud("world/clip_text_hybrid (green)", points[idx_hybrid],
                        np.tile([[0.2, 1.0, 0.4]], (len(idx_hybrid), 1)), radii=0.025)
            
            if "features_dino" in data:
                print("Filtering with DINO...")
                dino_feat = data["features_dino"].cpu().numpy()
                min_samples = 2 #max(2, len(idx_hybrid) // 10)
                idx_dino_filtered = filter_with_dino_dbscan(
                    points, dino_feat, idx_hybrid,
                    eps=0.4, min_samples=min_samples, spatial_weight=0.3
                )
                print(f"✅ DINO DBSCAN retained {len(idx_dino_filtered)} / {len(idx_hybrid)} points")
                log_pointcloud("world/clip_text_hybrid_dino (purple)", points[idx_dino_filtered],
                            np.tile([[0.7, 0.3, 1.0]], (len(idx_dino_filtered), 1)), radii=0.03)

            


        # Run CLI query if given, then continue into interactive loop
        if args.query:
            print(f"🔍 One-shot query: '{args.query}'")
            run_query(args.query)

        print("\n💬 Enter additional queries (or 'q' to quit):")
        while True:
            try:
                user_input = input("🔍 Query: ").strip()
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
