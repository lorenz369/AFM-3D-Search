import torch
import rerun as rr
import numpy as np
from sklearn.cluster import DBSCAN
import open3d as o3d
from matplotlib import cm
import argparse
import matplotlib.pyplot as plt



def log_pointcloud(name, points, colors, radii=0.015):
    rr.log(name, rr.Points3D(positions=points, colors=colors, radii=radii))

def normalize(arr):
    return (arr - arr.min(axis=0)) / (arr.max(axis=0) - arr.min(axis=0) + 1e-8)

def cluster_dino(points, dino_features, eps=0.3, min_samples=10, spatial_weight=0.5):
    dino_norm = dino_features / (np.linalg.norm(dino_features, axis=1, keepdims=True) + 1e-8)
    spatial_norm = normalize(points)
    combined = np.concatenate([dino_norm, spatial_norm * spatial_weight], axis=1)
    cluster_labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(combined)
    return cluster_labels

def visualize_dino_clusters(points, labels):
    min_cluster_size = 30

    # Filter out small clusters and noise
    large_clusters = {l for l in np.unique(labels) if l != -1 and (labels == l).sum() >= min_cluster_size}
    labels = np.array([l if l in large_clusters else -1 for l in labels])

    # Setup colormap
    cmap = plt.get_cmap("gist_ncar")
    label_max = max(large_clusters) if large_clusters else 1

    # Optional: Dim or skip noise
    noise_mask = labels == -1
    if np.any(noise_mask):
        log_pointcloud("world/cluster_noise", points[noise_mask], np.tile([[0.2, 0.2, 0.2]], (np.sum(noise_mask), 1)), radii=0.005)

    # Log each cluster separately
    for label in sorted(large_clusters):
        mask = labels == label
        cluster_color = np.tile(cmap(label / label_max)[:3], (np.sum(mask), 1))
        log_pointcloud(f"world/cluster_{label}", points[mask], cluster_color, radii=0.01)

    print(f"✅ Visualized {len(large_clusters)} clusters (separately)")


def main(args):
    rr.init("DINO 3D Segmentation", spawn=True)
    data = torch.load(args.pt_path, map_location='cpu')
    points = data["points"].numpy()
    dino_feat = data["features_dino"].cpu().numpy()

    print("📊 Clustering with DINO + spatial features...")
    labels = cluster_dino(points, dino_feat, eps=0.4, min_samples=3, spatial_weight=0.5)
    visualize_dino_clusters(points, labels)
    print(f"Cluster label counts: {np.unique(labels, return_counts=True)}")


    if args.ply_path:
        print(f"📎 Logging original PLY: {args.ply_path}")
        pcd = o3d.io.read_point_cloud(args.ply_path)
        points_ply = np.asarray(pcd.points)
        colors_ply = np.asarray(pcd.colors) if pcd.has_colors() else np.ones_like(points_ply) * 0.8
        log_pointcloud("world/original_pointcloud", points_ply, colors_ply, radii=0.01)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("pt_path", help="Path to .pt file with points and features_dino")
    parser.add_argument("--ply-path", help="Optional original .ply for reference")
    args = parser.parse_args()
    main(args)