import torch
import rerun as rr
import numpy as np
from sklearn.decomposition import PCA
import clip
import torch.nn.functional as F
import argparse
import open3d as o3d

# ---------- Helper functions ----------
def pca_to_rgb(features):
    pca = PCA(n_components=3).fit_transform(features)
    pca = (pca - pca.min(axis=0)) / (pca.max(axis=0) - pca.min(axis=0) + 1e-6)
    print(f"PCA stats → min: {pca.min(axis=0)}, max: {pca.max(axis=0)}")

    return pca

def log_feature_stats(name, features):
    rr.log(f"stats/{name}_dim", rr.TextDocument(f"Dim: {features.shape[1]}"))
    norms = np.linalg.norm(features, axis=1)
    rr.log(f"stats/{name}_norm_mean", rr.TextDocument(f"Mean L2 norm: {norms.mean():.3f}"))
    rr.log(f"stats/{name}_norm_std", rr.TextDocument(f"Std L2 norm: {norms.std():.3f}"))

def log_pointcloud(name, points, colors):
    rr.log(name, rr.Points3D(positions=points, colors=colors))

def text_query_clip(points, features_clip, query_text, model, device):
    with torch.no_grad():
        text_feat = model.encode_text(clip.tokenize([query_text]).to(device))
        text_feat = F.normalize(text_feat, dim=-1)
        features_clip_np = features_clip if isinstance(features_clip, np.ndarray) else features_clip.cpu().numpy()
        features_clip_np = PCA(n_components=512).fit_transform(features_clip_np)
        features_clip = torch.tensor(features_clip_np).to(device)
        features_clip = F.normalize(features_clip, dim=-1)
        sim = (features_clip @ text_feat.T).squeeze().cpu().numpy()
    return sim

# ---------- Main ----------
def main(args):
    rr.init("Locate3D Diagnostics", spawn=True)

    # Load data
    # GPU: data = torch.load(args.pt_path)
    data = torch.load(args.pt_path, map_location=torch.device('cpu'))
    points = data["points"].numpy()
    rgb = data["rgb"].numpy()
    if rgb.max() > 1.0: rgb = rgb / 255.0

    rr.log("world/points_rgb", rr.Points3D(points, colors=rgb))

    # Feature visualizations
    for feat_key in ["features_clip", "features_dino", "features_sam"]:
        if feat_key in data:
            feat = data[feat_key].cpu().numpy().astype(np.float32)
            colors = pca_to_rgb(feat)
            log_pointcloud(f"world/{feat_key}_pca", points, colors)
            log_feature_stats(feat_key, feat)

    # CLIP text query highlighting (optional)
    if "features_clip" in data and args.query:
        print(f"Running CLIP query for: '{args.query}'")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, _ = clip.load("ViT-B/32", device=device)
        sim = text_query_clip(points, data["features_clip"], args.query, model, device)
        topk = sim.argsort()[-args.topk:]
        rr.log("world/clip_text_match", rr.Points3D(points[topk], radii=0.015, colors=[[1,0,0]]*len(topk)))
        rr.log("stats/clip_text", rr.TextDocument(f"Top {args.topk} points for: {args.query}"))

    # Log basic stats
    rr.log("stats/num_points", rr.TextDocument(f"Total points: {len(points):,}"))
    bounds = f"""X: {points[:,0].min():.2f} → {points[:,0].max():.2f}
Y: {points[:,1].min():.2f} → {points[:,1].max():.2f}
Z: {points[:,2].min():.2f} → {points[:,2].max():.2f}"""
    rr.log("stats/bounding_box", rr.TextDocument(bounds))

    norm_sim = (sim - sim.min()) / (sim.max() - sim.min() + 1e-6)
    heat_colors = np.stack([norm_sim, np.zeros_like(norm_sim), 1 - norm_sim], axis=1)
    rr.log("world/clip_text_heatmap", rr.Points3D(points, colors=heat_colors, radii=0.008))




    # Load the .ply file
    ply_path = "locate-3d/locate3d_data/dataset/arkitscenes/raw/Training/42444821/42444821_3dod_mesh.ply"  # Change this to your file path
    pcd = o3d.io.read_point_cloud(ply_path)

    # Get positions
    points = np.asarray(pcd.points)

    # Optional: get colors (if available)
    if pcd.has_colors():
        colors = np.asarray(pcd.colors)
    else:
        colors = np.ones_like(points) * 0.8  # Gray fallback

    # Optional: get normals
    # normals = np.asarray(pcd.normals) if pcd.has_normals() else None

    # Log to Rerun
    rr.log("world/ply_points", rr.Points3D(positions=points, colors=colors, radii=0.01))

# ---------- CLI ----------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("pt_path", help="Path to Locate3D .pt file")
    parser.add_argument("--query", type=str, help="CLIP text prompt (e.g. 'chair')")
    parser.add_argument("--topk", type=int, default=1000, help="Top-k points for text query (default: 1000)")
    args = parser.parse_args()
    main(args)
