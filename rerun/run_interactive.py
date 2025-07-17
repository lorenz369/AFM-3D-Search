#!/usr/bin/env python3
"""Hydra entry-point for the Interactive Text-Search visualiser.

    python run_interactive.py pointcloud_dir=/data/my_scene
"""
import hydra
from omegaconf import DictConfig
from omegaconf import OmegaConf

from src.visualize_featurized_pointcloud import discover_featurized_files
from src.visualize_interactive_text_search import InteractiveTextSearch


@hydra.main(version_base="1.3", config_path="config", config_name="base_config")
def main(cfg: DictConfig) -> None:
    """
    Launch the interactive text-search viewer with parameters from *cfg*.
    If cfg.original_pointcloud is set, the original (PLY) pointcloud will also be visualized for reference.
    """

    print("\n================ Hydra Config Used ================")
    print(OmegaConf.to_yaml(cfg))
    print("==================================================\n")

    # ---------------------------------------------------------------------
    # 1. Locate point-cloud (.pt) files
    try:
        files_info = discover_featurized_files(cfg.pointcloud_dir)
        featurized_available = True
    except Exception as e:
        print(f"[Warning] Could not load featurized pointcloud: {e}")
        files_info = None
        featurized_available = False

    # ---------------------------------------------------------------------
    # 2. Spin up the Interactive viewer
    if featurized_available:
        viewer = InteractiveTextSearch(
            files_info,
            file_key=cfg.file_type,
            create_mesh=cfg.rendering.create_mesh,
            outlier_method=cfg.interactive_search.outlier_method,
            use_statistical_outliers=cfg.interactive_search.use_statistical_outliers,
            use_dino_filtering=cfg.interactive_search.use_dino_filtering,
            original_pointcloud=getattr(cfg, "original_pointcloud", None),
            config=cfg,
        )
        # ---------------------------------------------------------------------
        # 3. Serve the Rerun stream on the configured port
        # ---------------------------------------------------------------------
        port = cfg.server.port if "server" in cfg and "port" in cfg.server else 9878
        mode = cfg.server.mode if "server" in cfg and "mode" in cfg.server else "local"
        viewer.run_interactive_session(mode=mode, port=port)
    else:
        # Only visualize the original pointcloud if provided
        if getattr(cfg, "original_pointcloud", None):
            print("[Info] Only visualizing the original pointcloud (no featurized data available)")
            # Minimal rerun session to show the original pointcloud
            import rerun as rr
            from src.visualize_featurized_pointcloud import load_original_pointcloud
            rr.init("Original_Pointcloud_Only", spawn=True)
            rr.spawn(port=9878)
            rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Y_UP, static=True)
            try:
                points, colors = load_original_pointcloud(cfg.file_type, cfg.original_pointcloud)
                if colors is not None:
                    rr.log("world/original_pointcloud", rr.Points3D(points, colors=colors, radii=0.008), static=True)
                else:
                    rr.log("world/original_pointcloud", rr.Points3D(points, radii=0.008), static=True)
                print("Original pointcloud logged to rerun.")
            except Exception as e:
                print(f"[Warning] Could not visualize original pointcloud: {e}")
            print("[Info] No interactive text search available without featurized data.")
        else:
            print("[Error] No valid featurized pointcloud or original pointcloud provided. Nothing to visualize.")


if __name__ == "__main__":
    main() 