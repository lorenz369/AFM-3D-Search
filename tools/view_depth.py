import argparse
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description="View a depth map from a .npz file.")
parser.add_argument('file_path', type=str, help="Path to the .npz file to display.")
args = parser.parse_args()

# Load the .npz file using the path from the command line
try:
    with np.load(args.file_path) as data:
        depth_map = data['depth']

    print(f"File: {args.file_path}")
    print(f"Shape of the depth map: {depth_map.shape}")
    print(f"Minimum distance: {depth_map.min():.2f} meters")
    print(f"Maximum distance: {depth_map.max():.2f} meters")
    print(f"Average distance: {depth_map.mean():.2f} meters")

    center_y, center_x = depth_map.shape[0] // 2, depth_map.shape[1] // 2
    print(f"Distance at the center of the image: {depth_map[center_y, center_x]:.2f} meters")


    plt.imshow(depth_map, cmap='viridis')
    plt.colorbar(label='Distance (meters)')
    plt.title(f'Depth Map for {args.file_path}')
    plt.show()

except FileNotFoundError:
    print(f"Error: The file '{args.file_path}' was not found.")
    print("Please make sure you have the correct file path.")
except KeyError:
    print(f"Error: The file '{args.file_path}' does not contain a 'depth' array.")
