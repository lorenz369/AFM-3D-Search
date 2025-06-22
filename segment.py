from segment_anything import SamPredictor, sam_model_registry
import cv2
import numpy as np
import torch
import os
from tqdm import tqdm



# Load the SAM model
sam = sam_model_registry["vit_h"](checkpoint="sam_vit_h.pth")
predictor = SamPredictor(sam)
sam.to(torch.device("cuda"))


# Function to run SAM on a single image
def run_sam_on_image(image_path):
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    predictor.set_image(image_rgb)
    masks, _, _ = predictor.predict()
    return masks, image_rgb

# Liste aller Frame-Dateien
frame_dir = "data/frames/AFM_Video_Marco_1"

# Process all frames in the folder
for i in tqdm(range(len(os.listdir(frame_dir))), desc="Segmenting frames"):
    masks, image = run_sam_on_image(f"{frame_dir}/frame_{i:04d}.png")
    
    # Convert boolean mask to uint8 (0 or 255) before saving
    mask_uint8 = (masks[0].astype(np.uint8)) * 255
    
    # Save the mask and the RGB image
    cv2.imwrite(f"data/masks/AFM_Video_Marco_1/mask_{i:04d}.png", mask_uint8)
    cv2.imwrite(f"data/images/AFM_Video_Marco_1/image_{i:04d}.png", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))