import rerun as rr
import open3d as o3d
import numpy as np
import os
import cv2
from pathlib import Path

rr.init("AFM_Video_Viewer", spawn=True)

# --- Settings ---
base_path = Path("MASt3R-SLAM/logs/AFM_Video_Marco_1")
pcd_path = base_path / "AFM_Video_Marco_1.ply"
pose_file = base_path / "AFM_Video_Marco_1_with_intrinsics.txt"
keyframe_dir = base_path / "keyframes"
depth_dir = base_path / "depth_maps"
log_root = Path("world")

# --- Coordinate system & point cloud ---
rr.log(str(log_root), rr.ViewCoordinates.RDF, static=True)

pcd = o3d.io.read_point_cloud(str(pcd_path))
pcd = pcd.voxel_down_sample(0.02)
points = np.asarray(pcd.points)
colors = np.asarray(pcd.colors) if pcd.has_colors() else np.ones_like(points)
rr.log(f"{log_root}/point_cloud", rr.Points3D(positions=points, colors=colors), static=True)

# --- Load poses ---
camera_poses = {}
with open(pose_file, "r") as f:
    for line in f:
        values = list(map(float, line.strip().split()))
        frame_id = int(values[0])
        tx, ty, tz = values[1:4]
        qx, qy, qz, qw = values[4:8]  # note order
        fx, fy, cx, cy = values[8:12]
        camera_poses[frame_id] = {
            "translation": np.array([tx, ty, tz], dtype=np.float32),
            "rotation_quat": np.array([qw, qx, qy, qz], dtype=np.float32),
            "intrinsics": (fx, fy, cx, cy),
        }

# --- Load keyframes & depth maps ---
keyframe_files = sorted([f for f in os.listdir(keyframe_dir) if f.endswith(".png")])
depth_files = sorted([f for f in os.listdir(depth_dir) if f.endswith(".npy")])
num_frames = min(len(keyframe_files), len(depth_files))

for i in range(num_frames):
    rr.set_time_sequence("frame", i)

    frame_id = i
    image_path = keyframe_dir / keyframe_files[i]
    depth_path = depth_dir / depth_files[i]

    # Load and log image
    image = cv2.imread(str(image_path))
    h, w = image.shape[:2]
    rr.log(f"{log_root}/camera/pinhole/image", rr.Image(image, color_model=rr.ColorModel.BGR))

    # Load and log depth
    depth = np.load(str(depth_path))
    # Normalize to 0–255 and convert to uint8
    if np.max(depth) > 0:
        depth_normalized = (depth / np.max(depth) * 255).astype(np.uint8)
    else:
        depth_normalized = np.zeros_like(depth, dtype=np.uint8)

    # Ensure it's 1 channel (CV_8UC1)
    if len(depth_normalized.shape) == 2:
        depth_normalized = depth_normalized

    # Apply OpenCV color map to make it BGR
    depth_colormap = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)

    # Log it as color image under correct time frame
    rr.log(f"{log_root}/camera/depth_colored", rr.Image(depth_colormap, color_model=rr.ColorModel.BGR))
 

    # Camera transform
    if frame_id in camera_poses:
        pose = camera_poses[frame_id]
        translation = pose["translation"]
        quat = pose["rotation_quat"]

        rr.log(f"{log_root}/camera", rr.Transform3D(
            translation=translation.tolist(),
            rotation=quat.tolist()  # [w, x, y, z]
        ))

        # Camera intrinsics (only once)
        if i == 0:
            fx, fy, cx, cy = pose["intrinsics"]
            rr.log(f"{log_root}/camera/pinhole", rr.Pinhole(
                resolution=[w, h],
                focal_length=[fx, fy],
                principal_point=[cx, cy],
                camera_xyz=rr.ViewCoordinates.RUB
            ))
