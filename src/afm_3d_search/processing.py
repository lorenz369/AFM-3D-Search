import torch
import numpy as np
import argparse

def save_to_ply(filepath, xyz, rgb):
    # (This function is unchanged)
    if not filepath.endswith('.ply'):
        filepath += '.ply'
    xyz = np.asarray(xyz, dtype=np.float32)
    rgb = np.asarray(rgb, dtype=np.uint8)
    header = [
        "ply", "format ascii 1.0", f"element vertex {len(xyz)}",
        "property float x", "property float y", "property float z",
        "property uchar red", "property uchar green", "property uchar blue",
        "end_header"
    ]
    with open(filepath, 'w') as f:
        for line in header:
            f.write(line + '\n')
        for i in range(len(xyz)):
            f.write(f"{xyz[i, 0]} {xyz[i, 1]} {xyz[i, 2]} {rgb[i, 0]} {rgb[i, 1]} {rgb[i, 2]}\n")
    print(f"✅ Successfully saved point cloud to: {filepath}")

def main():
    parser = argparse.ArgumentParser(
        description="Inspect a .pt file and extract its point cloud to a .ply file."
    )
    parser.add_argument("pt_file", type=str, help="Path to the input .pt cache file.")
    args = parser.parse_args()

    # --- 1. Inspect the .pt file ---
    print(f"🔎 Inspecting file: {args.pt_file}\n")
    try:
        data = torch.load(args.pt_file)
    except Exception as e:
        print(f"❌ Error loading file: {e}")
        return

    print("File contains the following keys:")
    for key in data.keys():
        value = data[key]
        if isinstance(value, torch.Tensor):
            print(f"  - {key}: (Tensor, shape={value.shape})")
        else:
            print(f"  - {key}: (Type: {type(value)})")
    
    print("-" * 30)

    # --- 2. Extract and save the point cloud (using CORRECT keys) ---
    if 'points' in data and data['points'] is not None:
        print("Found point cloud data. Extracting...")
        
        xyz_points = data['points'].cpu().numpy()
        rgb_colors = data['rgb'].cpu().numpy()

        if rgb_colors.dtype == np.float32 or rgb_colors.dtype == np.float64:
            rgb_colors = (rgb_colors * 255).astype(np.uint8)

        output_ply_file = args.pt_file.replace('.pt', '.ply')
        save_to_ply(output_ply_file, xyz_points, rgb_colors)
            
    else:
        # --- 3. Provide detailed debugging steps if the point cloud is None ---
        print("❌ Point cloud data in the file is empty ('None').")
        print("This almost always means there was an issue with your custom depth maps.\n")
        print("💡 DEBUGGING STEPS:\n")
        print("1.  Check the Depth Scale Factor: Your depth maps from 'ml-depth-pro'")
        print("    might not need scaling. Open `locate-3d/locate3d_data/arkitscenes_dataset.py`")
        print("    and find the `DEPTH_SCALE_FACTOR`. Try changing it from `0.001` to `1.0`.")
        print("\n    `self.DEPTH_SCALE_FACTOR = 1.0`  <-- Try this\n")
        print("2.  After changing the scale factor, delete the old cache file:")
        print(f"    `rm {args.pt_file}`")
        print("    Then re-run the `run_preprocessing.py` script.\n")
        print("3.  If that doesn't work, your depth maps might be all zeros. You can")
        print("    verify this by temporarily adding a print statement in `arkitscenes_dataset.py`")
        print("    inside the depth loading loop (around line 120) to inspect a tensor:")
        print("\n    `print('Depth min/max:', depth.min(), depth.max())`\n")

if __name__ == "__main__":
    main()