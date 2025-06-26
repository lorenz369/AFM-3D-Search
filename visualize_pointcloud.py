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
import glob
from PIL import Image
from scipy.spatial.transform import Rotation

def discover_files(base_dir):
    """Auto-discover all required files from the top-level directory."""
    base_dir = os.path.abspath(base_dir)
    
    # Find the PLY file to determine the base name
    ply_files = glob.glob(os.path.join(base_dir, "*.ply"))
    if not ply_files:
        raise FileNotFoundError(f"No PLY file found in {base_dir}")
    
    ply_path = ply_files[0]
    base_name = os.path.splitext(os.path.basename(ply_path))[0]
    
    # Construct all paths
    poses_file = os.path.join(base_dir, f"{base_name}.txt")
    intrinsics_file = os.path.join(base_dir, f"{base_name}_with_intrinsics.txt")
    keyframes_dir = os.path.join(base_dir, "keyframes")
    depth_dir = os.path.join(base_dir, "depth_maps")
    
    # Verify all required files exist
    missing_files = []
    if not os.path.exists(poses_file):
        missing_files.append(poses_file)
    if not os.path.exists(intrinsics_file):
        missing_files.append(intrinsics_file)
    if not os.path.exists(keyframes_dir):
        missing_files.append(keyframes_dir)
    if not os.path.exists(depth_dir):
        missing_files.append(depth_dir)
    
    if missing_files:
        print(f"Warning: Missing files/directories: {missing_files}")
    
    print(f"Discovered files in {base_dir}:")
    print(f"  PLY: {ply_path}")
    print(f"  Poses: {poses_file}")
    print(f"  Intrinsics: {intrinsics_file}")
    print(f"  Keyframes: {keyframes_dir}")
    print(f"  Depth maps: {depth_dir}")
    
    return {
        'ply_path': ply_path,
        'poses_file': poses_file,
        'intrinsics_file': intrinsics_file,
        'keyframes_dir': keyframes_dir,
        'depth_dir': depth_dir,
        'base_name': base_name
    }

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
    Returns data indexed by frame_id for consistent mapping with keyframes/depth.
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

    # 2. Read intrinsics and frame info
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
        if len(parts) >= 17:  # Updated for new format without depth paths
            frame_idx = int(parts[0])
            tx, ty, tz, qx, qy, qz, qw = [float(p) for p in parts[1:8]]
            fx, fy, cx, cy = [float(p) for p in parts[8:12]]
            
            intrinsics_data.append({
                'frame_idx': frame_idx,
                'translation': np.array([tx, ty, tz]),
                'rotation_quat': np.array([qx, qy, qz, qw]),
                'fx': fx, 'fy': fy, 'cx': cx, 'cy': cy
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
        
        # Get intrinsics
        fx, fy = intrinsics_entry['fx'], intrinsics_entry['fy']
        cx, cy = intrinsics_entry['cx'], intrinsics_entry['cy']
        
        # Estimate reasonable image size if not available (common default)
        width, height = 640, 480
        
        # If cx, cy are zero, use image center as a reasonable default
        if cx == 0.0 and cy == 0.0:
            cx, cy = width / 2.0, height / 2.0
            print(f"Warning: Zero principal point detected. Using image center: ({cx}, {cy})")
        
        # If focal lengths seem too small, use reasonable defaults
        if fx < 100 or fy < 100:
            fx = fy = max(width, height) * 0.8  # Reasonable estimate
            print(f"Warning: Small focal length detected. Using estimate: fx={fx}, fy={fy}")
            
        frame_id = intrinsics_entry['frame_idx']
        cameras[frame_id] = {
            'timestamp': timestamp,
            'translation': np.array([tx, ty, tz]),
            'rotation_quat': np.array([qx, qy, qz, qw]),
            'rotation_matrix': rotation.as_matrix(),
            'intrinsics': np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]]),
            'resolution': (width, height),
            'frame_idx': frame_id
        }

    print(f"Successfully associated {len(cameras)} camera entries.")
    return cameras

def visualize_slam_data(
    files_info,
    mode="serve",
    remote_host=None, 
    remote_port=9876
):
    """Visualize complete SLAM data."""
    
    print(f"=== Starting SLAM Visualization ===")
    print(f"Mode: {mode}")
    print(f"Remote host: {remote_host}")
    print(f"Remote port: {remote_port}")
    
    rr.init("AFM_3D_SLAM_Visualization", spawn=False)
    print("✓ Rerun SDK initialized")

    if mode == "save":
        print("📁 Using SAVE mode - will write to output.rrd")
        rr.save("output.rrd")
    elif mode == "serve":
        print("🌐 Using SERVE mode")
        if remote_host:
            print(f"🔗 Remote mode: connecting to gRPC at {remote_host}:{remote_port}")
            rr.serve_grpc(grpc_port=remote_port)
        else:
            print("🖥️  Local mode: spawning Rerun viewer window")
            rr.spawn()
    else:
        print(f"❌ Unknown mode: {mode}")
        
    print("✓ Viewer setup complete")

    # Set up a static reference frame
    print("🌍 Setting up world coordinate frame...")
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Y_UP, static=True)

    # Log the main point cloud
    print("☁️  Loading and logging point cloud...")
    points, colors = read_ply_file(files_info['ply_path'])
    # Log the point cloud as timeless so it is visible regardless of the active timeline
    if colors is not None:
        print(f"✓ Logging {len(points)} points with colors (static)")
        rr.log("world/point_cloud", rr.Points3D(points, colors=colors), static=True)
    else:
        print(f"✓ Logging {len(points)} points without colors (static)")
        rr.log("world/point_cloud", rr.Points3D(points), static=True)

    # Load combined camera data (indexed by frame_id)
    print("📷 Loading camera data...")
    camera_data = read_camera_data(files_info['poses_file'], files_info['intrinsics_file'])
    sorted_frame_ids = sorted(camera_data.keys())
    print(f"✓ Found {len(camera_data)} camera frames to process")

    # Log camera poses, keyframes, and depth maps over time
    print("🎬 Processing camera frames...")
    for i, frame_id in enumerate(sorted_frame_ids):
        print(f"  Processing frame {i+1}/{len(sorted_frame_ids)}: {frame_id}")
        cam_info = camera_data[frame_id]
        
        # Use timestamp for the timeline
        rr.set_time("timeline", timestamp=cam_info['timestamp'])
        
        # Log camera transform
        rr.log(
            f"world/camera",
            rr.Transform3D(
                translation=cam_info['translation'], 
                mat3x3=cam_info['rotation_matrix']
            )
        )

        # Log keyframe image using frame_id (new consistent naming)
        keyframe_path = os.path.join(files_info['keyframes_dir'], f"{frame_id:06d}.png")
        if os.path.exists(keyframe_path):
            try:
                rr.log(
                    "world/camera/image", 
                    rr.Image(np.array(Image.open(keyframe_path))).compress(jpeg_quality=80)
                )
                print(f"    ✓ Loaded keyframe: {frame_id:06d}.png")
            except Exception as e:
                print(f"    ❌ Error loading keyframe {keyframe_path}: {e}")
        else:
            print(f"    ⚠️  Keyframe not found: {keyframe_path}")
        
        # Log depth map using frame_id (new consistent naming)
        depth_path = os.path.join(files_info['depth_dir'], f"{frame_id:06d}.npy")
        if os.path.exists(depth_path):
            try:
                depth_map = np.load(depth_path)
                rr.log(
                    "world/camera/depth",
                    rr.DepthImage(depth_map)
                )
                print(f"    ✓ Loaded depth map: {frame_id:06d}.npy, shape: {depth_map.shape}")
            except Exception as e:
                print(f"    ❌ Error loading depth map {depth_path}: {e}")
        else:
            print(f"    ⚠️  Depth map not found: {depth_path}")

        # Log camera intrinsics
        rr.log(
            "world/camera/image",
            rr.Pinhole(
                image_from_camera=cam_info['intrinsics'],
                resolution=cam_info['resolution']
            )
        )

    print("🎉 Visualization data logged to Rerun successfully!")
    if remote_host:
        print(f"🌐 Streaming to {remote_host}:{remote_port}. Open the Rerun viewer and connect.")
    else:
        print("🖥️  Check the Rerun viewer window that should have opened.")

def main():
    parser = argparse.ArgumentParser(description="Visualize MASt3R SLAM results with Rerun")
    parser.add_argument("slam_dir", 
                       help="Top-level directory containing SLAM results (e.g., logs/depth_video2_5_sec_test)")
    parser.add_argument("--mode",
                       choices=["serve", "save"],
                       default="serve",
                       help="Set the visualization mode: 'serve' to stream, 'save' to file.")
    parser.add_argument("--remote-host",
                       help="Remote host IP for Rerun streaming")
    parser.add_argument("--remote-port", type=int, default=9876)
    
    args = parser.parse_args()

    # Auto-discover all files from the top-level directory
    try:
        files_info = discover_files(args.slam_dir)
    except Exception as e:
        print(f"Error discovering files: {e}")
        return

    visualize_slam_data(
        files_info,
        mode=args.mode,
        remote_host=args.remote_host, 
        remote_port=args.remote_port
    )

    if args.mode == "serve" and args.remote_host:
        print("\nVisualization server is running. Press Enter on the server terminal to exit.")
        input()

if __name__ == "__main__":
    main() 