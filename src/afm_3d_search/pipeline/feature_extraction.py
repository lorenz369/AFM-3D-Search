import torch
from omegaconf import DictConfig
from typing import List
from PIL import Image
import gc
import requests
from tqdm import tqdm
from torchvision import transforms
import numpy as np
from pathlib import Path

# --- Model Imports ---
import clip
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

def _download_file(url, destination: Path):
    print(f"📦 Downloading required model: {destination.name}...")
    destination.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(url, stream=True)
    response.raise_for_status()
    total_size = int(response.headers.get('content-length', 0))
    with open(destination, 'wb') as f, tqdm(total=total_size, unit='iB', unit_scale=True, desc=destination.name) as bar:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            bar.update(len(chunk))

class _ClipEncoder:
    def __init__(self, version, device):
        self.device = device
        self.model, self.preprocess = clip.load(version.replace("_", "/"), device=self.device, jit=False)
        self.model.eval()

    @torch.no_grad()
    def encode_image(self, image: Image.Image):
        processed_image = self.preprocess(image).unsqueeze(0).to(self.device)
        return self.model.encode_image(processed_image).float()

class _MaskEmbeddingFeatureImageGenerator:
    def __init__(self, mask_generator, image_text_encoder, device):
        self.mask_generator = mask_generator
        self.image_text_encoder = image_text_encoder
        self.cosine_similarity = torch.nn.CosineSimilarity(dim=-1)
        self.device = device
        self.feat_dim = self.image_text_encoder.model.visual.output_dim

    @torch.no_grad()
    def generate_features(self, pil_image: Image.Image):
        image_np = np.array(pil_image)
        masks = self.mask_generator.generate(image_np)
        masks = list(filter(lambda x: x["bbox"][2] * x["bbox"][3] != 0, masks))
        if not masks: 
            return torch.zeros(image_np.shape[0], image_np.shape[1], self.feat_dim, dtype=torch.half, device=self.device)

        with torch.cuda.amp.autocast(enabled=self.device.startswith("cuda")):
            global_feat = self.image_text_encoder.encode_image(pil_image)
            global_feat = torch.nn.functional.normalize(global_feat, dim=-1)

        outfeat = torch.zeros(image_np.shape[0], image_np.shape[1], self.feat_dim, dtype=torch.half, device=self.device)
        feat_per_roi, roi_nonzero_inds, similarity_scores = [], [], []

        for mask in masks:
            _x, _y, _w, _h = map(int, mask["bbox"])
            img_roi = pil_image.crop((_x, _y, _x + _w, _y + _h))
            if img_roi.size[0] == 0 or img_roi.size[1] == 0: continue
            
            roifeat = torch.nn.functional.normalize(self.image_text_encoder.encode_image(img_roi), dim=-1)
            feat_per_roi.append(roifeat)
            roi_nonzero_inds.append(torch.from_numpy(mask["segmentation"]).to(self.device))
            similarity_scores.append(self.cosine_similarity(global_feat, roifeat))

        if not feat_per_roi: return outfeat

        softmax_scores = torch.nn.functional.softmax(torch.cat(similarity_scores), dim=0)
        for i, mask_seg in enumerate(roi_nonzero_inds):
            weighted_feat = torch.nn.functional.normalize(softmax_scores[i] * global_feat + (1 - softmax_scores[i]) * feat_per_roi[i], dim=-1).half()
            outfeat[mask_seg] = weighted_feat
        return outfeat

class FeatureExtractor:
    def __init__(self, cfg: DictConfig, device: str):
        self.cfg = cfg.models
        self.device = device
        self.target_height = None
        self.target_width = None

        print(f"🦖 Initializing DINOv2 model ({self.cfg.dino.version})...")
        self.dinov2_model = torch.hub.load('facebookresearch/dinov2', self.cfg.dino.version, verbose=False).to(self.device).eval()
        
        print("📎 Initializing SAM and CLIP models...")
        WEIGHTS_DIR = Path("weights")
        sam_checkpoint_path = WEIGHTS_DIR / self.cfg.sam.checkpoint
        if not sam_checkpoint_path.exists():
            _download_file("https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth", sam_checkpoint_path)

        sam_model = sam_model_registry["vit_h"](checkpoint=sam_checkpoint_path).to(self.device)
        mask_generator = SamAutomaticMaskGenerator(sam_model)
        clip_encoder = _ClipEncoder(version=self.cfg.clip.version, device=self.device)
        self.clip_feature_generator = _MaskEmbeddingFeatureImageGenerator(mask_generator, clip_encoder, self.device)

    def _init_transforms(self, height, width):
        self.target_height = height
        self.target_width = width
        self.dino_transforms = transforms.Compose([
            transforms.Resize((self.target_height, self.target_width)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    @torch.no_grad()
    def run_batch(self, pil_batch: List[Image.Image]) -> dict:
        if self.target_height is None:
            self._init_transforms(pil_batch[0].height, pil_batch[0].width)

        # --- DINO Feature Extraction ---
        transformed_images = torch.stack([self.dino_transforms(p) for p in pil_batch]).to(self.device)
        features_dict = self.dinov2_model.forward_features(transformed_images)
        patch_features = features_dict['x_norm_patchtokens']
        
        B, N, D = patch_features.shape
        H_patch, W_patch = self.target_height // 14, self.target_width // 14
        
        feature_map_2d = patch_features.reshape(B, H_patch, W_patch, D).permute(0, 3, 1, 2)
        dino_features = torch.nn.functional.interpolate(
            feature_map_2d, size=(self.target_height, self.target_width), mode='bilinear', align_corners=False
        )

        # --- CLIP Feature Extraction ---
        clip_features_list = []
        for image in pil_batch:
            resized_image = image.resize((self.target_width, self.target_height))
            feature_tensor = self.clip_feature_generator.generate_features(resized_image)
            clip_features_list.append(feature_tensor)
        clip_features = torch.stack(clip_features_list)
        
        gc.collect()
        torch.cuda.empty_cache()
        
        return {
            "dino_features": dino_features, # Shape: (B, D, H, W)
            "clip_features": clip_features  # Shape: (B, H, W, D)
        }