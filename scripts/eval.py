import numpy as np
import json
import open3d as o3d
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
import os
from pathlib import Path
import hydra
from omegaconf import DictConfig, OmegaConf
import sys

# Add rerun/src and afm_3d_search to sys.path ONCE, cleanly
sys.path.insert(0, os.path.abspath('./rerun/src'))
sys.path.insert(0, os.path.abspath('./src/afm_3d_search'))

from clip_encoder import ClipEncoder
from visualize_interactive_text_search import dino_guided_clip_smoothing
from run_pipeline import main as run_pipeline

# Assuming you want to later log with rerun
import rerun as rr


def compute_iou(box1, box2):
    box1_min = np.min(box1, axis=-1)
    box1_max = np.max(box1, axis=-1)
    box2_min = np.min(box2, axis=-1)
    box2_max = np.max(box2, axis=-1)

    inter_min = np.maximum(box1_min, box2_min)
    inter_max = np.minimum(box1_max, box2_max)
    inter_dim = np.maximum(inter_max - inter_min, 0)
    inter_vol = np.prod(inter_dim)

    vol1 = np.prod(box1_max - box1_min)
    vol2 = np.prod(box2_max - box2_min)

    iou = inter_vol / (vol1 + vol2 - inter_vol + 1e-6)
    return iou


def points_to_aabb(points):
    min_corner = np.min(points, axis=0)
    max_corner = np.max(points, axis=0)
    box = np.stack([min_corner, max_corner], axis=-1)
    return box


def get_scene_paths(scene_id, data_root="./data/output/arkit_scenes"):
    """Get all relevant paths for a scene."""
    scene_dir = Path(data_root) / "raw_subsampled" / "Training" / scene_id
    completed_dir = Path(data_root) / "completed" / scene_id
    
    paths = {
        'scene_dir': scene_dir,
        'completed_dir': completed_dir,
        'ply': scene_dir / f"{scene_id}_3dod_mesh.ply",
        'dino_features': completed_dir / "dino_features.npy",
        'clip_features': completed_dir / "clip_features.npy",
        'point_cloud': completed_dir / "point_cloud.ply"
    }
    return paths


def ensure_scene_processed(scene_id, config_path="conf/config.yaml"):
    """Ensure a scene has been processed through the pipeline."""
    paths = get_scene_paths(scene_id)
    
    # Check if features already exist
    if paths['dino_features'].exists() and paths['clip_features'].exists():
        print(f"✅ Scene {scene_id} already processed")
        return True
        
    print(f"🔄 Processing scene {scene_id} through pipeline...")
    
    # Create a config for this scene
    cfg = {
        'scene_id': scene_id,
        'paths': {
            'data_root': str(paths['scene_dir'].parent.parent.parent),
            'raw_dir_name': 'raw_subsampled/Training',
            'completed_dir_name': 'completed'
        }
    }
    
    # Run the pipeline using hydra
    try:
        with hydra.initialize(config_path=os.path.dirname(config_path)):
            config = hydra.compose(config_name=os.path.basename(config_path))
            # Update config with our scene-specific settings
            config.scene_id = scene_id
            config.paths.data_root = str(paths['scene_dir'].parent.parent.parent)
            # Run the pipeline
            run_pipeline(config)
        print(f"✅ Successfully processed scene {scene_id}")
        return True
    except Exception as e:
        print(f"❌ Failed to process scene {scene_id}")
        print(f"Error: {e}")
        return False


def process_text_query(scene_id, text_query, clip_version="ViT-B/32", device="cuda"):
    """Process a text query for a scene using the pipeline's outputs."""
    paths = get_scene_paths(scene_id)
    
    # Load point cloud
    if not paths['point_cloud'].exists():
        print(f"❌ Processed point cloud not found for scene {scene_id}")
        return np.array([]), np.array([]).reshape(0, 3)
    
    pcd = o3d.io.read_point_cloud(str(paths['point_cloud']))
    points = np.asarray(pcd.points)
    
    # Load features
    try:
        dino_features = np.load(paths['dino_features'])
        clip_features = np.load(paths['clip_features'])
    except Exception as e:
        print(f"❌ Failed to load features for scene {scene_id}")
        print(f"Error: {e}")
        return np.array([]), np.array([]).reshape(0, 3)
    
    # Encode text query
    clip_encoder = ClipEncoder(version=clip_version, device=device)
    text_features = clip_encoder.encode_text(text_query).cpu().numpy()
    
    # Run CLIP DINOiser
    indices, matched_points, _, _, _, _, stats = dino_guided_clip_smoothing(
        points, clip_features, dino_features, text_features,
        threshold=0.22, k=10, mix_alpha=0.5
    )
    
    if len(indices) > 0:
        print(f"✅ Found {len(indices)} matching points")
        print(f"   Similarity range: {stats['min_sim']:.3f} - {stats['max_sim']:.3f}")
    else:
        print("⚠️  No matching points found")
    
    return indices, matched_points


def log_rerun_comparison(scene_id, points, predicted_indices, gt_boxes, query, run_dir="rerun_logs"):
    """Log the comparison of prediction and ground truth for a query using Rerun, and save to disk."""
    import os
    os.makedirs(run_dir, exist_ok=True)
    # Use a safe filename for the query
    import re
    safe_query = re.sub(r'[^a-zA-Z0-9_\-]', '_', query)[:40]
    recording_id = f"{scene_id}_{safe_query}"
    rr.init(recording_id, recording_id=recording_id, spawn=False)
    rr.set_time_sequence("query", 0)

    # Log the full point cloud (background)
    rr.log("pointcloud", rr.Points3D(points, colors=[0.7, 0.7, 0.7], radii=0.005))

    # Log predicted points (your method)
    if len(predicted_indices) > 0:
        rr.log("predicted", rr.Points3D(points[predicted_indices], colors=[1.0, 0.0, 0.0], radii=0.01))

    # Log ground truth boxes (as green corners)
    for i, gt_box in enumerate(gt_boxes):
        gt_points = np.array(gt_box)  # shape (8, 3) for a box
        rr.log(f"gt_box_{i}", rr.Points3D(gt_points, colors=[0.0, 1.0, 0.0], radii=0.012))

    # Save the log for later viewing
    save_path = os.path.join(run_dir, f"{recording_id}.rrd")
    rr.save(save_path)
    print(f"💾 Rerun log saved to {save_path}")


def evaluate_locate3d(scene_list, locate3d_json_path, output_dir):
    """Run evaluation on a list of scenes."""
    os.makedirs(output_dir, exist_ok=True)
    all_results = {}

    # Load annotations
    with open(locate3d_json_path, 'r') as f:
        annotations = json.load(f)
    annotations = [ann for ann in annotations if ann['scene_dataset'] == 'ARKitScenes']
    scenes_in_annotations = set(ann['scene_id'] for ann in annotations)

    # Process each scene
    for scene_id in scene_list:
        if scene_id not in scenes_in_annotations:
            print(f"⚠️  Skipping scene {scene_id}: no annotations found")
            continue
            
        print(f"\n🔍 Processing scene {scene_id}")
        print("=" * 50)
        
        # Ensure scene is processed
        if not ensure_scene_processed(scene_id):
            print(f"⚠️  Skipping scene {scene_id}: processing failed")
            continue
        
        # Get annotations for this scene
        scene_annotations = [ann for ann in annotations if ann['scene_id'] == scene_id]
        results = []
        
        # Process each query
        for ann in scene_annotations:
            description = ann['description']
            gt_boxes = ann['gt_boxes']
            
            print(f"\n📝 Query: '{description}'")
            predicted_indices, predicted_points = process_text_query(scene_id, description)
            
            if len(predicted_points) == 0:
                iou = 0.0
            else:
                pred_box = points_to_aabb(predicted_points)
                ious = [compute_iou(pred_box, np.array(gt_box)) for gt_box in gt_boxes]
                iou = max(ious)
                print(f"📊 IoU: {iou:.3f}")
            
            # Log rerun comparison for later viewing
            # Load the full point cloud for logging
            paths = get_scene_paths(scene_id)
            pcd = o3d.io.read_point_cloud(str(paths['point_cloud']))
            points = np.asarray(pcd.points)
            log_rerun_comparison(scene_id, points, predicted_indices, gt_boxes, description)

            results.append({
                'description': description,
                'iou': iou,
                'gt_boxes': gt_boxes,
                'num_predicted_points': len(predicted_points)
            })
        
        all_results[scene_id] = results
        
        # Save per-scene results
        result_path = Path(output_dir) / f'{scene_id}_results.json'
        with open(result_path, 'w') as f:
            json.dump(results, f, indent=4)
        print(f"\n💾 Saved results to {result_path}")

    # Save all results
    with open(Path(output_dir) / 'all_results.json', 'w') as f:
        json.dump(all_results, f, indent=4)

    # Print summary statistics
    all_ious = []
    for scene_results in all_results.values():
        all_ious.extend([r['iou'] for r in scene_results])
    
    if all_ious:
        print("\n📊 Final Results")
        print("=" * 50)
        print(f"Total queries: {len(all_ious)}")
        print(f"Mean IoU: {np.mean(all_ious):.3f}")
        print(f"Acc@0.25: {np.mean([iou > 0.25 for iou in all_ious]):.3f}")
        print(f"Acc@0.5: {np.mean([iou > 0.5 for iou in all_ious]):.3f}")


# Usage Example
if __name__ == "__main__":
    scene_ids = ['42445873']  # Add more scene IDs as needed
    
    print("🚀 Starting evaluation")
    print("=" * 50)
    
    evaluate_locate3d(
        scene_ids,
        'data/locate3d/arkit_scenes/train_arkitscenes.json',
        './output_results'
    )
