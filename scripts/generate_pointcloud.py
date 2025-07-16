import hydra
from omegaconf import DictConfig, OmegaConf
import os
import torch
import numpy as np
import trimesh
from PIL import Image
import gc
import requests
from tqdm import tqdm
from pathlib import Path

# --- Models and Utilities for Feature Extraction ---
import clip
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from torchvision import transforms
from typing import List

# ##################################################################################
# ## FEATURE EXTRACTION FUNCTIONS
# ##################################################################################

def _download_file(url, destination):
    """Internal helper to download model weights with a progress bar."""
    print(f"📦 Downloading required model: {url.split('/')[-1]}...")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    total_size = int(response.headers.get('content-length', 0))
    with open(destination, 'wb') as f, tqdm(
        total=total_size, unit='iB', unit_scale=True, unit_divisor=1024,
        desc=destination.split('/')[-1]
    ) as bar:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            bar.update(len(chunk))

def extract_dino_features(image_paths, dino_version, target_height, target_width, device, batch_size):
    """Extracts dense DINOv2 feature maps for a list of images in batches."""
    print(f"🦖 Initializing DINOv2 model ({dino_version})...")
    dinov2_model = torch.hub.load('facebookresearch/dinov2', dino_version, verbose=False).to(device).eval()

    dino_transforms = transforms.Compose([
        transforms.Resize((target_height, target_width)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    all_features = []
    print(f"🦖 Extracting DINOv2 features from {len(image_paths)} images...")
    with torch.no_grad():
        for i in tqdm(range(0, len(image_paths), batch_size), desc="DINOv2 Batches"):
            batch_paths = image_paths[i:i+batch_size]
            pil_images = [Image.open(p).convert("RGB") for p in batch_paths]
            transformed_images = torch.stack([dino_transforms(p) for p in pil_images]).to(device)

            features_dict = dinov2_model.forward_features(transformed_images)
            patch_features = features_dict['x_norm_patchtokens']

            B, N, D = patch_features.shape
            H_patch = target_height // 14
            W_patch = target_width // 14

            feature_map_2d = patch_features.reshape(B, H_patch, W_patch, D).permute(0, 3, 1, 2)
            upsampled_features = torch.nn.functional.interpolate(
                feature_map_2d, size=(target_height, target_width), mode='bilinear', align_corners=False
            )
            all_features.append(upsampled_features.permute(0, 2, 3, 1).cpu())
            torch.cuda.empty_cache()

    del dinov2_model
    gc.collect()
    print("✅ DINOv2 feature extraction complete.")
    return torch.cat(all_features, dim=0).numpy()

class _ClipEncoder:
    def __init__(self, version, device):
        self.device = device
        self.model, self.preprocess = clip.load(version.replace("_", "/"), device=self.device, jit=False)

    @torch.no_grad()
    def encode_image(self, image: np.ndarray):
        pil_image = Image.fromarray(image.astype(np.uint8))
        processed_image = self.preprocess(pil_image).unsqueeze(0).to(self.device)
        return self.model.encode_image(processed_image).float()

class _MaskEmbeddingFeatureImageGenerator:
    def __init__(self, mask_generator, image_text_encoder, device):
        self.mask_generator = mask_generator
        self.image_text_encoder = image_text_encoder
        self.cosine_similarity = torch.nn.CosineSimilarity(dim=-1)
        self.device = device
        self.feat_dim = self.image_text_encoder.model.visual.output_dim

    @torch.no_grad()
    def generate_features(self, image_np: np.ndarray):
        masks = self.mask_generator.generate(image_np)
        masks = list(filter(lambda x: x["bbox"][2] * x["bbox"][3] != 0, masks))
        if not masks: return torch.zeros(image_np.shape[0], image_np.shape[1], self.feat_dim, dtype=torch.half, device=self.device)

        with torch.cuda.amp.autocast(enabled=self.device.startswith("cuda")):
            global_feat = self.image_text_encoder.encode_image(image_np)
            global_feat = torch.nn.functional.normalize(global_feat, dim=-1)

        outfeat = torch.zeros(image_np.shape[0], image_np.shape[1], self.feat_dim, dtype=torch.half, device=self.device)
        feat_per_roi, roi_nonzero_inds, similarity_scores = [], [], []

        for mask in masks:
            _x, _y, _w, _h = map(int, mask["bbox"])
            img_roi = image_np[_y:_y+_h, _x:_x+_w]
            if img_roi.size == 0: continue
            roifeat = torch.nn.functional.normalize(self.image_text_encoder.encode_image(img_roi), dim=-1)
            feat_per_roi.append(roifeat)
            roi_nonzero_inds.append(torch.from_numpy(mask["segmentation"]))
            similarity_scores.append(self.cosine_similarity(global_feat, roifeat))

        if not feat_per_roi: return outfeat

        softmax_scores = torch.nn.functional.softmax(torch.cat(similarity_scores), dim=0)
        for i, mask_seg in enumerate(roi_nonzero_inds):
            weighted_feat = torch.nn.functional.normalize(softmax_scores[i] * global_feat + (1 - softmax_scores[i]) * feat_per_roi[i], dim=-1).half()
            outfeat[mask_seg] = weighted_feat
        return outfeat

def extract_clip_features(image_paths, clip_version, sam_checkpoint, target_height, target_width, device, batch_size):
    """Extracts dense, SAM-blended CLIP feature maps for a list of images."""
    print("📎 Initializing SAM and CLIP models...")
    if not os.path.exists(sam_checkpoint):
        _download_file("https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth", sam_checkpoint)

    sam_model = sam_model_registry["vit_h"](checkpoint=sam_checkpoint).to(device)
    mask_generator = SamAutomaticMaskGenerator(sam_model)
    clip_encoder = _ClipEncoder(version=clip_version, device=device)
    feature_generator = _MaskEmbeddingFeatureImageGenerator(mask_generator, clip_encoder, device)

    all_features = []
    print(f"📎 Extracting CLIP (SAM-blended) features from {len(image_paths)} images...")
    with torch.no_grad():
        for i in tqdm(range(0, len(image_paths), batch_size), desc="CLIP Batches"):
            batch_paths = image_paths[i:i+batch_size]
            pil_images = [Image.open(p).convert("RGB").resize((target_width, target_height)) for p in batch_paths]
            for pil_image in pil_images:
                all_features.append(feature_generator.generate_features(np.array(pil_image)).cpu())
            torch.cuda.empty_cache()

    del sam_model, mask_generator, clip_encoder, feature_generator
    gc.collect()
    print("✅ CLIP feature extraction complete.")
    return torch.stack(all_features, dim=0).numpy()

# ##################################################################################
# ## CORE SCRIPT LOGIC
# ##################################################################################


from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map


def aggregate_points_and_features(points, colors, features_dict, voxel_size):
    """Aggregates points, colors, and features that fall into the same voxel."""
    print(f"🧊 Voxelizing and aggregating points with voxel size {voxel_size}...")
    voxel_indices = np.floor(points / voxel_size).astype(int)

    voxel_data = {}
    for i in tqdm(range(len(voxel_indices)), desc="Mapping points to voxels"):
        voxel_key = tuple(voxel_indices[i])
        if voxel_key not in voxel_data:
            voxel_data[voxel_key] = {'points': [], 'colors': [], 'dino': [], 'clip': []}
        voxel_data[voxel_key]['points'].append(points[i])
        voxel_data[voxel_key]['colors'].append(colors[i])
        voxel_data[voxel_key]['dino'].append(features_dict['dino'][i])
        voxel_data[voxel_key]['clip'].append(features_dict['clip'][i])

    num_voxels = len(voxel_data)
    dino_dim = features_dict['dino'].shape[1]
    clip_dim = features_dict['clip'].shape[1]

    agg_points = np.zeros((num_voxels, 3), dtype=np.float32)
    agg_colors = np.zeros((num_voxels, 3), dtype=np.float32)
    agg_dino_features = np.zeros((num_voxels, dino_dim), dtype=np.float32)
    agg_clip_features = np.zeros((num_voxels, clip_dim), dtype=np.float32)

    for i, key in enumerate(tqdm(voxel_data.keys(), desc="Averaging voxel data")):
        data = voxel_data[key]
        agg_points[i] = np.mean(data['points'], axis=0)
        agg_colors[i] = np.mean(data['colors'], axis=0)
        agg_dino_features[i] = np.mean(data['dino'], axis=0)
        agg_clip_features[i] = np.mean(data['clip'], axis=0)

    print(f"✅ Aggregation complete. Original points: {len(points)}, Aggregated points: {num_voxels}")
    return agg_points, agg_colors, {'dino': agg_dino_features, 'clip': agg_clip_features}

def save_point_cloud(points, colors, filename):
    """Saves a colored point cloud to a .ply file."""
    if colors.max() <= 1.0: colors *= 255
    pc = trimesh.PointCloud(vertices=points, colors=colors.astype(np.uint8))
    pc.export(filename)
    print(f"✅ Point cloud saved successfully to {filename}")

@hydra.main(config_path="conf", config_name="config", version_base=None)
def generate_feature_cloud(cfg: DictConfig) -> None:
    """
    Main function to generate a 3D feature cloud, driven by Hydra configuration.
    """
    # --- Setup and Configuration ---
    print("--- Starting Point Cloud Generation Pipeline ---")
    print(OmegaConf.to_yaml(cfg))

    output_dir = Path(os.getcwd()) # Hydra automatically sets the working directory
    print(f"📁 Output directory for this run: {output_dir}")

    # Determine a base name for output files from the image folder
    image_folder_path = Path(cfg.paths.image_folder)
    base_name = image_folder_path.name
    output_ply_path = output_dir / f"{base_name}.ply"
    output_dino_npy = output_dir / f"{base_name}_dino_features.npy"
    output_clip_npy = output_dir / f"{base_name}_clip_features.npy"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" and torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    # --- VGGT 3D Reconstruction ---
    print(f"🔄 Initializing and loading VGGT model on {device}...")
    vggt_model = VGGT()
    vggt_model.load_state_dict(torch.hub.load_state_dict_from_url("https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt", map_location=device))
    vggt_model.eval().to(device)

    image_paths = sorted([str(p) for p in image_folder_path.glob('*') if p.suffix.lower() in ('.png', '.jpg', '.jpeg')])
    if not image_paths: raise ValueError(f"No images found in {cfg.paths.image_folder}")

    print("🚀 Running VGGT Inference...")
    images_tensor = load_and_preprocess_images(image_paths).to(device)
    with torch.no_grad(), torch.cuda.amp.autocast(dtype=dtype):
        predictions = vggt_model(images_tensor)

    print("✅ VGGT Inference complete.")
    B, S, C, H, W = predictions["images"].shape
    extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], (H, W))
    world_points = unproject_depth_map_to_point_map(
        predictions["depth"].squeeze(0).cpu().numpy(), extrinsic.squeeze(0).cpu().numpy(), intrinsic.squeeze(0).cpu().numpy()
    )
    confidence = predictions["depth_conf"].squeeze(0).cpu().numpy()
    images_np = predictions["images"].squeeze(0).cpu().numpy()

    print("🧹 Cleaning up VGGT model from GPU memory...")
    del vggt_model, images_tensor, predictions
    gc.collect(); torch.cuda.empty_cache()

    # --- Feature Extraction Steps ---
    dino_features = extract_dino_features(
        image_paths, cfg.models.dino_version, H, W, device, cfg.processing.dino_batch_size
    )
    clip_features = extract_clip_features(
        image_paths, cfg.models.clip_version, cfg.models.sam_checkpoint, H, W, device, cfg.processing.clip_batch_size
    )

    # --- Data Processing and Filtering ---
    points_flat = world_points.reshape(-1, 3)
    colors_flat = np.transpose(images_np, (0, 2, 3, 1)).reshape(-1, 3)
    confidence_flat = confidence.reshape(-1)
    
    dino_features_flat = dino_features.reshape(-1, dino_features.shape[-1])
    clip_features_flat = clip_features.reshape(-1, clip_features.shape[-1])

    print(f"🔍 Applying confidence filter (keeping top {100 - cfg.processing.conf_percentile}% of points)...")
    if cfg.processing.conf_percentile > 0:
        keep_mask = confidence_flat >= np.percentile(confidence_flat, cfg.processing.conf_percentile)
        filtered_points = points_flat[keep_mask]
        filtered_colors = colors_flat[keep_mask]
        filtered_dino_features = dino_features_flat[keep_mask]
        filtered_clip_features = clip_features_flat[keep_mask]
    else:
        filtered_points, filtered_colors = points_flat, colors_flat
        filtered_dino_features, filtered_clip_features = dino_features_flat, clip_features_flat
        
    print(f"✅ Filtering complete. Final point count: {len(filtered_points)}")

    # --- Voxel Aggregation ---
    if cfg.processing.voxel_size > 0:
        features_dict_unaggregated = {'dino': filtered_dino_features, 'clip': filtered_clip_features}
        agg_points, agg_colors, agg_features_dict = aggregate_points_and_features(
            filtered_points, filtered_colors, features_dict_unaggregated, cfg.processing.voxel_size
        )
    else:
        agg_points, agg_colors = filtered_points, filtered_colors
        agg_features_dict = {'dino': filtered_dino_features, 'clip': filtered_clip_features}

    # --- Save Outputs Separately ---
    print("💾 Saving final outputs...")
    
    save_point_cloud(agg_points, agg_colors, output_ply_path)
    
    np.save(output_dino_npy, agg_features_dict['dino'])
    print(f"✅ DINO features saved to {output_dino_npy}")
    
    np.save(output_clip_npy, agg_features_dict['clip'])
    print(f"✅ CLIP features saved to {output_clip_npy}")
    print("\n--- Pipeline Finished Successfully! ---")

if __name__ == "__main__":
    # This script is now a Hydra application.
    # The @hydra.main decorator handles parsing the configuration.
    generate_feature_cloud()