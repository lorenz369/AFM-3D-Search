from segment_anything import SamPredictor, sam_model_registry, SamAutomaticMaskGenerator
import cv2
import numpy as np
import torch
import os
from tqdm import tqdm
import json



# --- Setup paths ---
frame_dir = "data/frames/AFM_Video_Marco_1"
output_mask_dir = "data/segment/AFM_Video_Marco_1/masks/objects"
output_label_dir = "data/segment/AFM_Video_Marco_1/masks/labels"
output_meta_dir = "data/segment/AFM_Video_Marco_1/metadata"

# Create output folders
os.makedirs(output_mask_dir, exist_ok=True)
os.makedirs(output_label_dir, exist_ok=True)
os.makedirs(output_meta_dir, exist_ok=True)


# Load the SAM model
sam = sam_model_registry["vit_h"](checkpoint="sam_vit_h.pth")
predictor = SamPredictor(sam)
sam.to(torch.device("cuda"))


# Setup SAM Mask Generator
mask_generator = SamAutomaticMaskGenerator(
    model=sam,
    points_per_side=32,      # Erhöhen → mehr Details
    pred_iou_thresh=0.88,
    stability_score_thresh=0.95,
    min_mask_region_area=100, # filtert Mini-Masken raus
)


# Liste aller Frame-Dateien
frame_dir = "data/frames/AFM_Video_Marco_1"



for i in tqdm(range(len(os.listdir(frame_dir))), desc="Segmenting frames"):
    frame_index = int(f"frame_{i:04d}.png".split("_")[-1].split(".")[0])  # extract index from "frame_0000.png"

    # Load and prepare image
    image = cv2.imread(f"{frame_dir}/frame_{i:04d}.png")
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Generate masks
    masks = mask_generator.generate(image_rgb)

    # Create label map and metadata list
    label_map = np.zeros(image_rgb.shape[:2], dtype=np.uint8)
    metadata = []

    for j, m in enumerate(masks):
        # Save individual binary mask
        mask = (m["segmentation"].astype(np.uint8)) * 255
        mask_path = os.path.join(output_mask_dir, f"mask_{frame_index:04d}_{j:02d}.png")
        cv2.imwrite(mask_path, mask)

        # Add to label map with unique ID
        label_map[m["segmentation"]] = j + 1

        # Save object metadata
        metadata.append({
            "object_id": j + 1,
            "area": int(m["area"]),
            "bbox": m["bbox"],  # [x, y, width, height]
            "predicted_iou": float(m["predicted_iou"]),
            "stability_score": float(m["stability_score"]),
        })

    # Save label image
    label_path = os.path.join(output_label_dir, f"label_mask_{frame_index:04d}.png")
    cv2.imwrite(label_path, label_map)

    # Save metadata
    meta_path = os.path.join(output_meta_dir, f"frame_{frame_index:04d}.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
