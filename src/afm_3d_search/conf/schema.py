# src/afm_3d_search/conf/schema.py

from dataclasses import dataclass
from typing import List

@dataclass
class PathsConfig:
    run_output_dir: str # The directory for the main generation run
    output_dir: str     # The directory for validation scripts

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