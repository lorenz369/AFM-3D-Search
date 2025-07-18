import torch
import trimesh
import numpy as np
from pathlib import Path
from omegaconf import DictConfig
from torch_scatter import scatter_mean

@torch.no_grad()
def unproject_depth_to_points_gpu(depth, extr, intr):
    """
    Projects a batch of depth maps to 3D world coordinates using PyTorch on the GPU.
    depth: (B, H, W)
    extr: (B, 3, 4)
    intr: (B, 3, 3)
    """
    device = depth.device
    B, H, W = depth.shape
    
    ys, xs = torch.meshgrid(
        torch.arange(H, device=device, dtype=torch.float32),
        torch.arange(W, device=device, dtype=torch.float32),
        indexing="ij",
    )
    # ys, xs are (H, W), we need to expand for batch
    ys = ys.unsqueeze(0).expand(B, -1, -1)
    xs = xs.unsqueeze(0).expand(B, -1, -1)

    ones = torch.ones_like(xs)
    cam_coords = torch.stack([xs, ys, ones], axis=-1)  # (B, H, W, 3)
    
    # Apply depth
    cam_coords = cam_coords * depth.unsqueeze(-1)  # (B, H, W, 3)
    
    # Unproject using intrinsics
    # (B, H*W, 3) @ (B, 3, 3) -> (B, H*W, 3)
    cam_coords_flat = cam_coords.reshape(B, H * W, 3)
    inv_intr = torch.inverse(intr)
    points_cam = torch.bmm(cam_coords_flat, inv_intr.transpose(1, 2))
    
    # Transform to world coordinates using extrinsics
    hom_points_cam = torch.cat([points_cam, torch.ones(B, H * W, 1, device=device)], dim=-1) # (B, H*W, 4)
    
    # Create homogeneous extrinsic matrix (4x4)
    bottom_row = torch.tensor([0.0, 0.0, 0.0, 1.0], device=device).reshape(1, 1, 4).expand(B, -1, -1)
    extr_hom = torch.cat((extr, bottom_row), dim=1) # (B, 4, 4)
    
    inv_extr_hom = torch.inverse(extr_hom)
    
    world_coords_flat = torch.bmm(hom_points_cam, inv_extr_hom.transpose(1, 2)) # (B, H*W, 4)
    
    return world_coords_flat[:, :, :3].reshape(B, H, W, 3) # (B, H, W, 3)

@torch.no_grad()
def process_batch_gpu(vggt_batch_gpu: dict, features_batch_gpu: dict, proc_cfg: DictConfig) -> dict:
    """
    Processes a single batch of data entirely on the GPU.
    """
    # Reshape tensors to be (B, H, W, C)
    depth = vggt_batch_gpu["depth_tensor"].squeeze(2) # (B, H, W)
    confidence = vggt_batch_gpu["confidence_tensor"].squeeze(2) # (B, H, W)
    colors = vggt_batch_gpu["images_tensor"].permute(0, 2, 3, 1) # (B, H, W, 3)
    
    # Features already have the right channel order
    dino_features = features_batch_gpu["dino_features"].permute(0, 2, 3, 1) # (B, H, W, D_dino)
    clip_features = features_batch_gpu["clip_features"] # (B, H, W, D_clip)

    extr = vggt_batch_gpu["extrinsic_tensor"].squeeze(0) # (B, 3, 4)
    intr = vggt_batch_gpu["intrinsic_tensor"].squeeze(0) # (B, 3, 3)

    # Project depth to points on GPU
    world_points = unproject_depth_to_points_gpu(depth, extr, intr)

    # Flatten all tensors
    points_flat = world_points.reshape(-1, 3)
    colors_flat = colors.reshape(-1, 3)
    confidence_flat = confidence.reshape(-1)
    dino_flat = dino_features.reshape(-1, dino_features.shape[-1])
    clip_flat = clip_features.reshape(-1, clip_features.shape[-1])
    
    # --- Filtering on GPU ---
    if proc_cfg.conf_percentile > 0:
        conf_threshold = torch.quantile(confidence_flat.float(), proc_cfg.conf_percentile / 100.0)
        keep_mask = confidence_flat >= conf_threshold
        
        if not torch.any(keep_mask):
            return {"points": None, "colors": None, "dino_features": None, "clip_features": None}
            
        points_flat = points_flat[keep_mask]
        colors_flat = colors_flat[keep_mask]
        dino_flat = dino_flat[keep_mask]
        clip_flat = clip_flat[keep_mask]

    # --- Voxel Aggregation on GPU ---
    if proc_cfg.voxel_size > 0:
        voxel_indices = torch.floor(points_flat / proc_cfg.voxel_size).long()
        
        # Use unique to find the indices for scattering
        unique_voxels, inverse_indices = torch.unique(voxel_indices, dim=0, return_inverse=True)
        
        # Aggregate using torch_scatter
        agg_points = scatter_mean(points_flat, inverse_indices, dim=0)
        agg_colors = scatter_mean(colors_flat, inverse_indices, dim=0)
        agg_dino = scatter_mean(dino_flat, inverse_indices, dim=0)
        agg_clip = scatter_mean(clip_flat, inverse_indices, dim=0)
        
        return {
            "points": agg_points, "colors": agg_colors,
            "dino_features": agg_dino, "clip_features": agg_clip
        }
    else:
        return {
            "points": points_flat, "colors": colors_flat,
            "dino_features": dino_flat, "clip_features": clip_flat
        }

def save_artifacts(output_dir: Path, final_data_cpu: dict):
    """Saves the final NumPy arrays from CPU to disk."""
    print(f"💾 Saving final outputs to {output_dir}...")
    
    points_cpu = final_data_cpu['points']
    colors_cpu = final_data_cpu['colors']

    if points_cpu is None or len(points_cpu) == 0:
        print("No points to save. Skipping artifact generation.")
        return

    ply_path = output_dir / "point_cloud.ply"
    
    # Ensure colors are in the correct format for trimesh
    if colors_cpu.max() <= 1.0:
        colors_cpu = (colors_cpu * 255)
    colors_cpu = colors_cpu.astype(np.uint8)

    pc = trimesh.PointCloud(vertices=points_cpu, colors=colors_cpu)
    pc.export(ply_path)
    print(f"✅ Point cloud saved to {ply_path}")

    np.save(output_dir / "dino_features.npy", final_data_cpu['dino_features'])
    print(f"✅ DINO features saved")
    
    np.save(output_dir / "clip_features.npy", final_data_cpu['clip_features'])
    print(f"✅ CLIP features saved")