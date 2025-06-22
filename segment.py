from segment_anything import SamPredictor, sam_model_registry
import cv2
import numpy as np
import torch
import os

sam = sam_model_registry["vit_h"](checkpoint="sam_vit_h.pth")
predictor = SamPredictor(sam)

def run_sam_on_image(image_path):
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    predictor.set_image(image_rgb)
    masks, _, _ = predictor.predict()
    return masks, image_rgb

for i in range(len(os.listdir("data/frames/AFM_Video_Marco_1"))):
    masks, image = run_sam_on_image(f"data/frames/AFM_Video_Marco_1/frame_{i:04d}.png")
    cv2.imwrite(f"data/masks/AFM_Video_Marco_1/mask_{i:04d}.png", masks[0])
    cv2.imwrite(f"data/images/AFM_Video_Marco_1/image_{i:04d}.png", image)