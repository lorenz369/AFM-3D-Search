#!/usr/bin/env python3
"""Hydra entry-point for the Interactive Text-Search visualiser.

    python run_interactive.py pointcloud_dir=/data/my_scene
"""
import hydra
from omegaconf import DictConfig

from visualize_featurized_pointcloud import discover_featurized_files
from visualize_interactive_text_search import InteractiveTextSearch


@hydra.main(version_base="1.3", config_path="config", config_name="interactive")
def main(cfg: DictConfig) -> None:
    """Launch the interactive text-search viewer with parameters from *cfg*."""

    # ---------------------------------------------------------------------
    # 1. Locate point-cloud (.pt) files
    # ---------------------------------------------------------------------
    files_info = discover_featurized_files(cfg.pointcloud_dir)

    # ---------------------------------------------------------------------
    # 2. Spin up the Interactive viewer
    # ---------------------------------------------------------------------
    viewer = InteractiveTextSearch(
        files_info,
        file_key=cfg.file_type,
        create_mesh=cfg.rendering.create_mesh,
        outlier_method=cfg.interactive_search.outlier_method,
        use_statistical_outliers=cfg.interactive_search.use_statistical_outliers,
        use_dino_filtering=cfg.interactive_search.use_dino_filtering,
    )

    # ---------------------------------------------------------------------
    # 3. Serve the Rerun stream on the configured port
    # ---------------------------------------------------------------------
    port = cfg.server.port if "server" in cfg and "port" in cfg.server else 9878
    viewer.run_interactive_session(port=port)


if __name__ == "__main__":
    main() 