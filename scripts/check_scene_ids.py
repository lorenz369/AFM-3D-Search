import os
import json

# 1. List all scene IDs in the folder
folder_path = "outputs/arkit_scenes/raw_subsampled/Training"
folder_scene_ids = set(os.listdir(folder_path))

# 2. List all scene IDs in the JSON
json_path = "data/locate3d/arkit_scenes/train_arkitscenes.json"
with open(json_path, "r") as f:
    annotations = json.load(f)
json_scene_ids = set(ann["scene_id"] for ann in annotations if "scene_id" in ann)

# 3. Compute overlap and differences
overlap = folder_scene_ids & json_scene_ids
only_in_folder = folder_scene_ids - json_scene_ids
only_in_json = json_scene_ids - folder_scene_ids

print(f"Total in folder: {len(folder_scene_ids)}")
print(f"Total in JSON: {len(json_scene_ids)}")
print(f"Overlap: {len(overlap)}")
print(f"Only in folder: {len(only_in_folder)}")
print(f"Only in JSON: {len(only_in_json)}")

print("\nSample overlap:", list(overlap)[:10])
print("\nSample only in folder:", list(only_in_folder)[:10])
print("\nSample only in JSON:", list(only_in_json)[:10])

# Save results to a text file
with open("scene_id_overlap_report.txt", "w") as out:
    out.write(f"Total in folder: {len(folder_scene_ids)}\n")
    out.write(f"Total in JSON: {len(json_scene_ids)}\n")
    out.write(f"Overlap: {len(overlap)}\n")
    out.write(f"Only in folder: {len(only_in_folder)}\n")
    out.write(f"Only in JSON: {len(only_in_json)}\n\n")
    out.write("Overlap IDs:\n" + "\n".join(sorted(overlap)) + "\n\n")
    out.write("Only in folder IDs:\n" + "\n".join(sorted(only_in_folder)) + "\n\n")
    out.write("Only in JSON IDs:\n" + "\n".join(sorted(only_in_json)) + "\n")
    print("\nResults saved to scene_id_overlap_report.txt") 