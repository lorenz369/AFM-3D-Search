import numpy as np
import os
import matplotlib.pyplot as plt

# Path to your depth maps directory
depth_maps_dir = 'MASt3R-SLAM/logs/AFM_Video_Marco_1/depth_maps'

# Thresholds for outlier detection (adjust as needed)
min_valid_depth = 0.1    # usually depths below this are invalid
max_valid_depth = 100.0  # upper bound for plausible depth

for filename in sorted(os.listdir(depth_maps_dir)):
    if filename.endswith('.npy'):
        filepath = os.path.join(depth_maps_dir, filename)
        depth_map = np.load(filepath)

        # Check for invalid values
        invalid_mask = (depth_map < min_valid_depth) | (depth_map > max_valid_depth) | np.isnan(depth_map)
        invalid_ratio = np.sum(invalid_mask) / depth_map.size

        print(f"{filename}:")
        print(f"  Min depth: {np.min(depth_map):.3f}, Max depth: {np.max(depth_map):.3f}")
        print(f"  Invalid values ratio: {invalid_ratio:.3%}")

        # Optionally visualize suspicious maps
        if invalid_ratio > 0.05:  # Arbitrary threshold
            plt.imshow(depth_map, cmap='plasma')
            plt.title(f"{filename} (invalid ratio: {invalid_ratio:.2%})")
            plt.colorbar()
            plt.show()