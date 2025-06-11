#!/usr/bin/env python3
"""
Pointcloud visualization script using Rerun SDK
Visualizes the PLY pointcloud from MASt3R results
"""

import rerun as rr
import numpy as np
from plyfile import PlyData, PlyElement
import argparse
import os
from pathlib import Path
from PIL import Image

def read_ply_file(ply_path):
    """Read PLY file and extract points and colors"""
    plydata = PlyData.read(ply_path)
    
    # Debug: print the structure of the PLY file
    print(f"PLY elements: {[elem.name for elem in plydata.elements]}")
    
    vertex = plydata['vertex']
    print(f"Vertex element type: {type(vertex)}")
    
    # Get the actual data from the PlyElement
    vertex_data = vertex.data
    print(f"Vertex data type: {type(vertex_data)}")
    print(f"Vertex data shape: {vertex_data.shape if hasattr(vertex_data, 'shape') else 'No shape'}")
    print(f"Vertex dtype: {vertex_data.dtype}")
    print(f"Vertex field names: {vertex_data.dtype.names}")
    
    # Extract coordinates using field names
    field_names = vertex_data.dtype.names
    print(f"Available fields: {field_names}")
    
    # Try different common field names for coordinates
    x_field = 'x' if 'x' in field_names else ('X' if 'X' in field_names else field_names[0])
    y_field = 'y' if 'y' in field_names else ('Y' if 'Y' in field_names else field_names[1])
    z_field = 'z' if 'z' in field_names else ('Z' if 'Z' in field_names else field_names[2])
    
    print(f"Using coordinate fields: {x_field}, {y_field}, {z_field}")
    
    points = np.column_stack((vertex_data[x_field], vertex_data[y_field], vertex_data[z_field]))
    print(f"Extracted {len(points)} points")
    
    # Extract colors if available
    colors = None
    color_fields = ['red', 'green', 'blue']
    alt_color_fields = ['r', 'g', 'b']
    
    if all(field in field_names for field in color_fields):
        colors = np.column_stack((vertex_data['red'], vertex_data['green'], vertex_data['blue']))
        print("Found RGB color fields")
    elif all(field in field_names for field in alt_color_fields):
        colors = np.column_stack((vertex_data['r'], vertex_data['g'], vertex_data['b']))
        print("Found r/g/b color fields")
    else:
        print("No color fields found")
    
    # Normalize colors if they're in 0-255 range
    if colors is not None and colors.max() > 1.0:
        colors = colors / 255.0
        print("Normalized colors from 0-255 to 0-1 range")
    
    return points, colors

def visualize_pointcloud(ply_path, keyframes_dir=None):
    """Visualize pointcloud and optionally keyframe images"""
    
    # Initialize Rerun
    rr.init("AFM 3D Pointcloud Visualization", spawn=True)
    
    # Read and log the pointcloud
    print(f"Loading pointcloud from: {ply_path}")
    points, colors = read_ply_file(ply_path)
    
    print(f"Loaded {len(points)} points")
    
    # Log the pointcloud
    if colors is not None:
        rr.log("pointcloud", rr.Points3D(points, colors=colors))
    else:
        rr.log("pointcloud", rr.Points3D(points))
    
    # If keyframes directory is provided, log the images
    if keyframes_dir and os.path.exists(keyframes_dir):
        print(f"Loading keyframes from: {keyframes_dir}")
        
        # Get all PNG files and sort them by timestamp
        keyframe_files = sorted([f for f in os.listdir(keyframes_dir) if f.endswith('.png')])
        
        for i, filename in enumerate(keyframe_files):
            filepath = os.path.join(keyframes_dir, filename)
            # Extract timestamp from filename (the number before .png)
            timestamp = float(filename.replace('.png', ''))
            
            # Load the image using PIL
            try:
                img = Image.open(filepath)
                img_array = np.array(img)
                
                # Log the image with timestamp using correct API
                rr.set_time_seconds("timeline", timestamp)
                rr.log(f"keyframes/{filename}", rr.Image(img_array))
                
                if i % 10 == 0:  # Progress indicator
                    print(f"Loaded {i+1}/{len(keyframe_files)} keyframes")
                    
            except Exception as e:
                print(f"Warning: Could not load image {filename}: {e}")
                continue
    
    print("Visualization complete! Check the Rerun viewer.")

def main():
    parser = argparse.ArgumentParser(description="Visualize PLY pointcloud with Rerun")
    parser.add_argument("--ply", default="data/mast3r_results/AFM_Video_Marco_1.ply", 
                       help="Path to PLY file")
    parser.add_argument("--keyframes", default="data/mast3r_results/keyframes/AFM_Video_Marco_1",
                       help="Path to keyframes directory")
    parser.add_argument("--no-keyframes", action="store_true", 
                       help="Skip loading keyframes")
    
    args = parser.parse_args()
    
    # Check if files exist
    if not os.path.exists(args.ply):
        print(f"Error: PLY file not found: {args.ply}")
        return
    
    keyframes_dir = None if args.no_keyframes else args.keyframes
    if keyframes_dir and not os.path.exists(keyframes_dir):
        print(f"Warning: Keyframes directory not found: {keyframes_dir}")
        keyframes_dir = None
    
    visualize_pointcloud(args.ply, keyframes_dir)

if __name__ == "__main__":
    main() 