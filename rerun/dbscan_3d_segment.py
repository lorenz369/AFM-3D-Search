import torch
import numpy as np
import rerun as rr
import argparse
from sklearn.cluster import DBSCAN
import open3d as o3d
from matplotlib import pyplot as plt

# ---- Utils ----
def log_pointcloud(name, points, colors, radii=0.01):
    rr.log(name, rr.Points3D(positions=points, colors=colors, radii=radii))


def main(args):
    rr.init("Geometric Segmentation", spawn=True)

    # Load point cloud
    data = torch.load(args.pt_path, map_location='cpu')
    points = data['points'].numpy()
    rgb = data['rgb'].numpy() if 'rgb' in data else np.ones_like(points) * 0.8
    features = data['features_clip'].numpy() if 'features_clip' in data else None

    # --- Normals via Open3D ---
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30))
    normals = np.asarray(pcd.normals)

    print(f"Point cloud scale: min {points.min(0)}, max {points.max(0)}")

    # --- DBSCAN on (xyz, normals) ---
    features_for_clustering = np.hstack((points, normals))
    db = DBSCAN(eps=0.2, min_samples=5).fit(features_for_clustering)
    labels = db.labels_

    num_clusters = labels.max() + 1 if labels.size > 0 else 0
    print(f"✅ Found {num_clusters} geometric instances via DBSCAN")

    # --- Visualization ---
    cmap = plt.get_cmap("tab20")
    for i in range(num_clusters):
        mask = labels == i
        color = np.tile(cmap(i % 20)[:3], (mask.sum(), 1))
        log_pointcloud(f"world/instance_{i}", points[mask], color, radii=0.01)

    if np.any(labels == -1):
        log_pointcloud("world/instance_noise", points[labels == -1],
                       np.tile([[0.3, 0.3, 0.3]], (np.sum(labels == -1), 1)),
                       radii=0.005)

    # Optional CLIP matching
    if features is not None and args.query:
        import clip
        import torch.nn.functional as F

        device = "cuda" if torch.cuda.is_available() else "cpu"
        clip_model, _ = clip.load("ViT-B/32", device=device)

        def match_clip(features, query_text):
            with torch.no_grad():
                text_feat = clip_model.encode_text(clip.tokenize([query_text]).to(device))
                text_feat = F.normalize(text_feat, dim=-1)
                features = torch.tensor(features).to(device)
                features = F.normalize(features, dim=-1)
                sim = (features @ text_feat.T).squeeze().cpu().numpy()
            return sim

        pooled_feats = []
        label_map = []
        for i in range(num_clusters):
            mask = labels == i
            if mask.sum() > 0:
                pooled_feats.append(features[mask].mean(axis=0))
                label_map.append(i)

        sims = match_clip(np.stack(pooled_feats), args.query)
        top_idx = np.argsort(sims)[::-1][:5]

        for rank, i in enumerate(top_idx):
            label = label_map[i]
            mask = labels == label
            log_pointcloud(f"world/clip_match_{rank}_{label}",
                           points[mask], np.tile([[1, 0.2, 0.2]], (mask.sum(), 1)), radii=0.02)

        rr.log("world/stats", rr.TextDocument(
            f"Top segments for: '{args.query}'\n" +
            "\n".join([f"Instance {label_map[i]} → sim={sims[i]:.2f}" for i in top_idx])))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("pt_path", help="Path to .pt pointcloud file")
    parser.add_argument("--query", type=str, help="Text query for CLIP match")
    args = parser.parse_args()
    main(args)
