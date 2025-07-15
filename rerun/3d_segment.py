import torch
import numpy as np
from pathlib import Path
import sys
sys.path.append("/Users/samihaddouti/code/AFM-3D-Search/Mask3D/models")
from mask3d_inference import Mask3DInference
sys.path.append("/Users/samihaddouti/code/AFM-3D-Search/Mask3D")
from models.model_factory import build_model


def run_mask3d_on_pt_file(pt_path, config_path, checkpoint_path):
    # Load your voxelized .pt data from Locate3D
    data = torch.load(pt_path, map_location='cpu')

    points = data['points'].numpy()          # [N, 3], meters
    rgb = data['rgb'].numpy()                # [N, 3], normalized [0, 1]
    features_clip = data['features_clip'].numpy()  # [N, D], for later CLIP matching

    # Prepare tensors for Mask3D
    points_tensor = torch.tensor(points, dtype=torch.float32)
    colors_tensor = torch.tensor(rgb, dtype=torch.float32)

    # Initialize Mask3D model
    model = Mask3DInference(config_path, checkpoint_path)

    # Run inference
    outputs = model.predict(points_tensor, colors_tensor)
    instance_labels = outputs["instance_labels"]  # [N,]

    print(f"✅ Found {instance_labels.max() + 1} instances")

    # Return outputs for further CLIP matching downstream
    return {
        "points": points,
        "rgb": rgb,
        "features_clip": features_clip,
        "instance_labels": instance_labels
    }


# ---- Example Usage ----
if __name__ == "__main__":
    pt_path = "locate-3d/cache/ARKitScenes/42444821_combined.pt"
    config_path = "/Users/samihaddouti/code/AFM-3D-Search/Mask3D/conf/data/datasets/scannet.yaml"
    checkpoint_path = "/Users/samihaddouti/code/AFM-3D-Search/Mask3D/checkpoints/scannet_val.ckpt"

    results = run_mask3d_on_pt_file(pt_path, config_path, checkpoint_path)

    # Example: Print number of points per instance
    instance_labels = results["instance_labels"]
    unique_labels, counts = np.unique(instance_labels, return_counts=True)
    print("\nInstances and their sizes:")
    for label, count in zip(unique_labels, counts):
        print(f"Instance {label}: {count} points")
