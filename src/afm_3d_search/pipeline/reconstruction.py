import torch
from PIL import Image
from typing import List, Dict


from vggt.models.vggt import VGGT
from vggt.utils.load_fn import _get_camera_transforms
from vggt.utils.pose_enc import pose_encoding_to_extri_intri


def run_vggt(pil_images: List[Image.Image], device: str, dtype: torch.dtype) -> Dict:
    """
    Runs VGGT reconstruction and returns a dictionary of raw GPU tensors.
    """
    print(f"🔄 Initializing VGGT model on {device}...")
    # In a persistent worker, this model would be cached to avoid reloading
    vggt_model = VGGT()
    vggt_model.load_state_dict(torch.hub.load_state_dict_from_url("https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt", map_location=device))
    vggt_model.eval().to(device)

    # Preprocess images to create a tensor
    transform = _get_camera_transforms(zoom_crop_transform=False)
    images_tensor = torch.stack([transform(img) for img in pil_images]).unsqueeze(0).to(device)

    print("🚀 Running VGGT Inference...")
    with torch.no_grad(), torch.cuda.amp.autocast(dtype=dtype):
        predictions = vggt_model(images_tensor)

    print("✅ VGGT Inference complete.")
    B, S, C, H, W = predictions["images"].shape
    extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], (H, W))

    return {
        "depth_tensor": predictions["depth"],
        "confidence_tensor": predictions["depth_conf"],
        "images_tensor": predictions["images"], # This contains the colors
        "extrinsic_tensor": extrinsic,
        "intrinsic_tensor": intrinsic,
        "height": H,
        "width": W
    }