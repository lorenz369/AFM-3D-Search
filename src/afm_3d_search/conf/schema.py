# src/afm_3d_search/conf/schema.py

from dataclasses import dataclass
from typing import List, Optional # FIX: 'Optional' is now imported

@dataclass
class PathsConfig:
    image_folder: str 
    output_dir: str

@dataclass
class ProcessingConfig:
    conf_percentile: float
    voxel_size: float
    dino_batch_size: int
    clip_batch_size: int

@dataclass
class ModelsConfig:
    clip_version: str
    sam_checkpoint: str
    dino_version: str

@dataclass
class Config:
    paths: PathsConfig
    processing: ProcessingConfig
    models: ModelsConfig
    run_name: str
    
    # FIX: These parameters are now officially part of the configuration
    scene_id: Optional[str] = None
    data_source: str = "staging"