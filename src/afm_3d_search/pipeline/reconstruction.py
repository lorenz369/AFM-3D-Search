import torch
from PIL import Image
from typing import List, Dict
import torchvision.transforms.functional as TF
import numpy as np
import gc
import torchvision.transforms.functional as TF
import numpy as np

from vggt.models.vggt import VGGT
from vggt.utils.pose_enc import pose_encoding_to_extri_intri



def _preprocess_single_image(img: Image.Image, mode: str = "crop", target_size: int = 518) -> torch.Tensor:
    """Applies the specific preprocessing steps from the original project to a single PIL image."""
    
    # If there's an alpha channel, blend onto a white background
    if img.mode == "RGBA":
        background = Image.new("RGBA", img.size, (255, 255, 255, 255))
        img = Image.alpha_composite(background, img).convert("RGB")
    elif img.mode != "RGB":
        img = img.convert("RGB")

    width, height = img.size
    
    # Calculate new dimensions, ensuring they are divisible by 14
    if mode == "pad":
        if width >= height:
            new_width = target_size
            new_height = round(height * (new_width / width) / 14) * 14
        else:
            new_height = target_size
            new_width = round(width * (new_height / height) / 14) * 14
    else:  # mode == "crop"
        new_width = target_size
        new_height = round(height * (new_width / width) / 14) * 14

    img = img.resize((new_width, new_height), Image.Resampling.BICUBIC)
    img_tensor = TF.to_tensor(img)  # Convert to tensor (0, 1)

    if mode == "crop" and new_height > target_size:
        img_tensor = TF.center_crop(img_tensor, [target_size, target_size])

    if mode == "pad":
        h_padding = target_size - img_tensor.shape[1]
        w_padding = target_size - img_tensor.shape[2]
        if h_padding > 0 or w_padding > 0:
            pad_top = h_padding // 2
            pad_bottom = h_padding - pad_top
            pad_left = w_padding // 2
            pad_right = w_padding - pad_left
            img_tensor = torch.nn.functional.pad(
                img_tensor, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=1.0
            )
            
    return img_tensor


# --- Main function for the pipeline ---
def run_vggt(pil_images: List[Image.Image], device: str, dtype: torch.dtype) -> Dict:
    """
    Runs VGGT reconstruction and returns a dictionary of raw GPU tensors.
    """
    print(f"🔄 Initializing VGGT model on {device}...")
    vggt_model = VGGT()
    vggt_model.load_state_dict(torch.hub.load_state_dict_from_url("https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt", map_location=device))
    vggt_model.eval().to(device)

    # Preprocess all images using our new helper function
    print("Pre-processing images with original project logic...")
    processed_images = [_preprocess_single_image(img, mode="crop") for img in pil_images]

    # Stacking logic to handle potentially different shapes after processing
    shapes = {img.shape for img in processed_images}
    if len(shapes) > 1:
        print(f"Warning: Found images with different shapes after processing: {shapes}. Padding to match.")
        max_height = max(shape[1] for shape in shapes)
        max_width = max(shape[2] for shape in shapes)
        
        padded_images = []
        for img in processed_images:
            h, w = img.shape[1], img.shape[2]
            h_padding = max_height - h
            w_padding = max_width - w
            if h_padding > 0 or w_padding > 0:
                pad_top = h_padding // 2
                pad_bottom = h_padding - pad_top
                pad_left = w_padding // 2
                pad_right = w_padding - pad_left
                img = torch.nn.functional.pad(img, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=1.0)
            padded_images.append(img)
        processed_images = padded_images

    # Create the final batch tensor for the model
    images_tensor = torch.stack(processed_images).unsqueeze(0).to(device)
    # Preprocess all images using our new helper function
    print("Pre-processing images with original project logic...")
    processed_images = [_preprocess_single_image(img, mode="crop") for img in pil_images]

    # Stacking logic to handle potentially different shapes after processing
    shapes = {img.shape for img in processed_images}
    if len(shapes) > 1:
        print(f"Warning: Found images with different shapes after processing: {shapes}. Padding to match.")
        max_height = max(shape[1] for shape in shapes)
        max_width = max(shape[2] for shape in shapes)
        
        padded_images = []
        for img in processed_images:
            h, w = img.shape[1], img.shape[2]
            h_padding = max_height - h
            w_padding = max_width - w
            if h_padding > 0 or w_padding > 0:
                pad_top = h_padding // 2
                pad_bottom = h_padding - pad_top
                pad_left = w_padding // 2
                pad_right = w_padding - pad_left
                img = torch.nn.functional.pad(img, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=1.0)
            padded_images.append(img)
        processed_images = padded_images

    # Create the final batch tensor for the model
    images_tensor = torch.stack(processed_images).unsqueeze(0).to(device)

    print("🚀 Running VGGT Inference...")
    with torch.no_grad(), torch.cuda.amp.autocast(dtype=dtype):
        predictions = vggt_model(images_tensor)

    print("✅ VGGT Inference complete.")
    B, S, C, H, W = predictions["images"].shape
    extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], (H, W))

    print("🧹 Cleaning up VGGT model from GPU memory...")
    del vggt_model, images_tensor
    gc.collect(); torch.cuda.empty_cache()

    return {
        "depth_tensor": predictions["depth"],
        "confidence_tensor": predictions["depth_conf"],
        "images_tensor": predictions["images"], 
        "extrinsic_tensor": extrinsic,
        "intrinsic_tensor": intrinsic,
        "height": H,
        "width": W
    }