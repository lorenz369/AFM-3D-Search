#!/usr/bin/env python3
"""
Pointcloud visualization script using Rerun SDK
Visualizes the PLY pointcloud from MASt3R results with remote streaming support
"""

import rerun as rr
import numpy as np
from plyfile import PlyData
import argparse
import os
from PIL import Image
from scipy.spatial.transform import Rotation

def read_ply_file(ply_path):
    """Read PLY file and extract points and colors."""
    plydata = PlyData.read(ply_path)
    vertex = plydata['vertex']
    points = np.column_stack([vertex['x'], vertex['y'], vertex['z']])
    
    colors = None
    if 'red' in vertex.data.dtype.names:
        colors = np.column_stack([vertex['red'], vertex['green'], vertex['blue']])
        if colors.max() > 1.0:
            colors = colors / 255.0
    
    print(f"Loaded {len(points)} points from {ply_path}")
    return points, colors

def read_camera_data(poses_file, intrinsics_file):
    """
    Read and combine camera pose and intrinsics data from two separate files.
    The poses file uses timestamps, while intrinsics file uses frame indices.
    """
    cameras = {}

    # 1. Read poses and timestamps
    if not os.path.exists(poses_file):
        print(f"Warning: Poses file not found: {poses_file}")
        return {}
        
    with open(poses_file, 'r') as f:
        poses_lines = f.readlines()

    poses_data = []
    for line in poses_lines:
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        
        parts = line.split()
        if len(parts) >= 8:
            timestamp = float(parts[0])
            pose_parts = [float(p) for p in parts[1:8]]  # tx, ty, tz, qx, qy, qz, qw
            poses_data.append((timestamp, pose_parts))

    # 2. Read intrinsics and depth map info
    if not os.path.exists(intrinsics_file):
        print(f"Warning: Intrinsics file not found: {intrinsics_file}")
        return {}

    with open(intrinsics_file, 'r') as f:
        intrinsics_lines = f.readlines()

    intrinsics_data = []
    for line in intrinsics_lines:
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        
        parts = line.split()
        if len(parts) >= 18:
            frame_idx = int(parts[0])
            tx, ty, tz, qx, qy, qz, qw = [float(p) for p in parts[1:8]]
            width, height = int(parts[8]), int(parts[9])
            fx, fy, cx, cy = [float(p) for p in parts[10:14]]
            depth_map_path = parts[17]
            
            intrinsics_data.append({
                'frame_idx': frame_idx,
                'translation': np.array([tx, ty, tz]),
                'rotation_quat': np.array([qx, qy, qz, qw]),
                'width': width,
                'height': height,
                'fx': fx, 'fy': fy, 'cx': cx, 'cy': cy,
                'depth_map_path': depth_map_path
            })

    # 3. Associate poses with intrinsics by index (assuming they correspond by order)
    print(f"Found {len(poses_data)} poses and {len(intrinsics_data)} intrinsics entries")
    
    min_entries = min(len(poses_data), len(intrinsics_data))
    for i in range(min_entries):
        timestamp, pose_parts = poses_data[i]
        intrinsics_entry = intrinsics_data[i]
        
        # Use pose data from poses file (more accurate)
        tx, ty, tz, qx, qy, qz, qw = pose_parts
        
        rotation = Rotation.from_quat([qx, qy, qz, qw])
        
        # Fix intrinsics if they look wrong (common issue)
        fx, fy = intrinsics_entry['fx'], intrinsics_entry['fy']
        cx, cy = intrinsics_entry['cx'], intrinsics_entry['cy']
        width, height = intrinsics_entry['width'], intrinsics_entry['height']
        
        # If cx, cy are zero, use image center as a reasonable default
        if cx == 0.0 and cy == 0.0:
            cx, cy = width / 2.0, height / 2.0
            print(f"Warning: Zero principal point detected. Using image center: ({cx}, {cy})")
        
        # If focal lengths seem too small, use reasonable defaults
        if fx < 100 or fy < 100:
            fx = fy = max(width, height) * 0.8  # Reasonable estimate
            print(f"Warning: Small focal length detected. Using estimate: fx={fx}, fy={fy}")
            
        cameras[timestamp] = {
            'translation': np.array([tx, ty, tz]),
            'rotation_quat': np.array([qx, qy, qz, qw]),
            'rotation_matrix': rotation.as_matrix(),
            'intrinsics': np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]]),
            'resolution': (width, height),
            'depth_map_path': intrinsics_entry['depth_map_path'],
            'frame_idx': intrinsics_entry['frame_idx']
        }

    print(f"Successfully associated {len(cameras)} camera entries.")
    return cameras

def visualize_slam_data(
    ply_path, 
    keyframes_dir, 
    poses_file, 
    intrinsics_file, 
    depth_dir,
    remote_host=None, 
    remote_port=9876
):
    """Visualize complete SLAM data."""
    
    rr.init("AFM_3D_SLAM_Visualization")

    if remote_host:
        rr.connect(addr=f"{remote_host}:{remote_port}")
    else:
        rr.spawn()

    # Set up a static reference frame
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Y_UP, static=True)

    # Log the main point cloud
    points, colors = read_ply_file(ply_path)
    if colors is not None:
        rr.log("world/point_cloud", rr.Points3D(points, colors=colors))
    else:
        rr.log("world/point_cloud", rr.Points3D(points))

    # Load combined camera data
    camera_data = read_camera_data(poses_file, intrinsics_file)
    sorted_timestamps = sorted(camera_data.keys())

    # Log camera poses, keyframes, and depth maps over time
    for timestamp in sorted_timestamps:
        rr.set_time_seconds("timeline", timestamp)
        cam_info = camera_data[timestamp]

        # Log camera transform
        rr.log(
            f"world/camera",
            rr.Transform3D(
                translation=cam_info['translation'], 
                mat3x3=cam_info['rotation_matrix']
            )
        )

        # Log keyframe image
        keyframe_path = os.path.join(keyframes_dir, f"{timestamp}.png")
        if os.path.exists(keyframe_path):
            rr.log(
                "world/camera/image", 
                rr.Image(keyframe_path).compress(jpeg_quality=80)
            )
        else:
            print(f"Warning: Keyframe not found at {keyframe_path}")
        
        # Log depth map
        # The depth map path in the intrinsics file is relative, need to fix the prefix
        depth_path = cam_info['depth_map_path']
        
        # Fix the path prefix if it starts with 'logs/' - should be 'MASt3R-SLAM/logs/'
        if depth_path.startswith('logs/'):
            depth_path = 'MASt3R-SLAM/' + depth_path
        
        if not os.path.isabs(depth_path):
            # If it's still a relative path, make it relative to the workspace root
            depth_path = os.path.join(os.getcwd(), depth_path)
        depth_path = os.path.normpath(depth_path)

        if os.path.exists(depth_path):
            try:
                depth_map = np.load(depth_path)
                rr.log(
                    "world/camera/depth",
                    rr.DepthImage(depth_map, from_parent=True)
                )
                print(f"Loaded depth map from {depth_path}, shape: {depth_map.shape}")
            except Exception as e:
                print(f"Error loading depth map from {depth_path}: {e}")
        else:
            print(f"Warning: Depth map not found at {depth_path}")

        # Log camera intrinsics
        rr.log(
            "world/camera/image",
            rr.Pinhole(
                image_from_camera=cam_info['intrinsics'],
                resolution=cam_info['resolution']
            )
        )

    print("Visualization data logged to Rerun.")
    if remote_host:
        print(f"Streaming to {remote_host}:{remote_port}. Open the Rerun viewer and connect.")

def main():
    parser = argparse.ArgumentParser(description="Visualize MASt3R SLAM results with Rerun")
    parser.add_argument("--ply", 
                       default="MASt3R-SLAM/logs/depth_video2_5_sec_fixed/AFM_Video_Marco_2_5_sec.ply")
    parser.add_argument("--keyframes", 
                       default="MASt3R-SLAM/logs/depth_video2_5_sec_fixed/keyframes/AFM_Video_Marco_2_5_sec")
    parser.add_argument("--poses", 
                       default="MASt3R-SLAM/logs/depth_video2_5_sec_fixed/AFM_Video_Marco_2_5_sec.txt")
    parser.add_argument("--intrinsics",
                       default="MASt3R-SLAM/logs/depth_video2_5_sec_fixed/AFM_Video_Marco_2_5_sec_with_intrinsics_and_depth.txt")
    parser.add_argument("--depth-dir",
                       default="MASt3R-SLAM/logs/depth_video2_5_sec_fixed/depth_maps")
    parser.add_argument("--remote-host",
                       help="Remote host IP for Rerun streaming")
    parser.add_argument("--remote-port", type=int, default=9876)
    
    args = parser.parse_args()

    visualize_slam_data(
        args.ply, 
        args.keyframes, 
        args.poses, 
        args.intrinsics,
        args.depth_dir,
        args.remote_host, 
        args.remote_port
    )

if __name__ == "__main__":
    main() 