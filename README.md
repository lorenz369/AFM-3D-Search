# AFM 3D Search — Natural-Language Object Search in 3D Scenes

Reconstruct a 3D scene from plain RGB video and search it with natural language ("red chair", "laptop on the desk") — training-free, built entirely from pretrained foundation models.

**Team:** Marco Lorenz ([@lorenz369](https://github.com/lorenz369)), Sami Haddouti ([@SamiHaddouti](https://github.com/SamiHaddouti)), Tim Cramer ([@tim-cramer](https://github.com/tim-cramer)) — developed in the Applied Foundation Models practical course at TUM.

## How it works

1. **Reconstruction** — [VGGT](https://github.com/facebookresearch/vggt) (git submodule) turns an RGB video into a dense 3D point cloud; dense SLAM (MASt3R-SLAM) and ARKit Visual-Inertial Odometry were evaluated as alternative reconstruction sources (`src/afm_3d_search/pipeline/reconstruction.py`).
2. **Featurization** — each point is enriched with CLIP semantics and DINO features, with SAM providing segmentation masks (`src/afm_3d_search/pipeline/feature_extraction.py`).
3. **Search** — a text query is embedded with CLIP and matched against the featurized point cloud; configurable outlier detection and DINO-based structural filtering sharpen the hits.
4. **Visualization** — interactive [Rerun](https://rerun.io/) viewer, either local or streamed from a remote GPU server (`rerun/`).

A FastAPI service with a job queue (`src/afm_3d_search/api/`, `worker.py`) wraps the pipeline for end-to-end scene processing. See [DEMO.md](DEMO.md) for a step-by-step walkthrough and [RERUN_demo.md](RERUN_demo.md) for the visualization demo.

## Table of Contents
- [Data](#data)
- [Setup](#setup)
- [Visualization of Point Clouds](#visualization-of-point-clouds)
- [Development Notes (TUM Infrastructure)](#development-notes-tum-infrastructure)

## Data
Large datasets and results are available in our [Google Drive folder](https://drive.google.com/drive/folders/184vJEGNb4RQ5tb9fF1LaFxy98oRriyPi?usp=drive_link).

## Setup

### Git Submodules
After cloning this repository, initialize and update the submodules:
```bash
# If you haven't cloned the repository yet:
git clone --recursive [repository-url]

# If you've already cloned the repository:
git submodule update --init --recursive

# To update submodules to their latest version:
git submodule update --remote
```

For the full environment setup (uv, dependencies, VGGT), follow [DEMO.md](DEMO.md).

## Visualization of Point Clouds

Interactive 3D visualization of 3D search results and featurized point clouds using the Rerun SDK.

### Environment Setup

#### Using uv
```bash
# Create and activate the virtual environment
uv venv .rerun_env --python 3.11
source .rerun_env/bin/activate

# Install required packages
uv pip install -r environments/rerun_requirements.txt
```

---

### Usage

#### **Interactive Text Search Visualization (Hydra-based)**

The main entry point for interactive semantic search on featurized point clouds is:

```bash
python rerun/run_interactive.py pointcloud_dir=<path/to/featurized/pt/files> file_type=<file_key>
```

- `pointcloud_dir`: Directory containing featurized `.pt` files (REQUIRED).
- `file_type`: Key or filename (without extension) of the `.pt` file to load (REQUIRED, e.g., `42447230`).
- You can override any config option via CLI, e.g.:
  - `rendering.create_mesh=true`
  - `interactive_search.outlier_method=iqr`
  - `interactive_search.use_dino_filtering=false`
  - `server.port=9878`
  - `server.mode=remote`
- The default config is in `rerun/config/base_config.yaml`. See that file for all options.

**Example:**
```bash
python rerun/run_interactive.py pointcloud_dir=../data/output/arkit_scenes/raw/featurized/ file_type=42447230
```

**Remote Mode & Port Forwarding:**

If you are running the script in `remote` server mode, you need to connect to the gRPC server from your local machine:

1. **Establish SSH port forwarding:**
   ```bash
   ssh -L {port}:localhost:{port} {user}@{server-ip}
   ```
   Replace `{port}` with the port number (e.g., 9878), `{user}` with your username, and `{server-ip}` with the server address.

2. **Connect to the gRPC server locally:**
   After starting the script in remote mode and establishing port forwarding, run:
   ```bash
   rerun --connect localhost:{port}/proxy
   ```
   This will connect your local Rerun viewer to the remote gRPC server via the forwarded port.

**Controls (in the terminal window):**
- Type search queries and press Enter (e.g., `chair`, `red sofa`)
- Type `clear` to remove highlights
- Type `q` to quit
- Type `help` for more commands and options (e.g., change outlier method, enable/disable DINO filtering, set thresholds)

**Advanced CLI overrides:**
```bash
python rerun/run_interactive.py pointcloud_dir=... file_type=... interactive_search.outlier_method=percentile rendering.create_mesh=true
```

#### **Script Options (Hydra config keys):**

| Option                                 | Description                                               | Default (base_config.yaml)         |
|-----------------------------------------|-----------------------------------------------------------|------------------------------------|
| `pointcloud_dir`                        | Path to featurized pointcloud `.pt` files                 | `../data/output/arkit_scenes/raw/featurized/` |
| `file_type`                             | Key/filename (no extension) of `.pt` file to load         | `"42447230"`                      |
| `original_pointcloud`                   | Path to original PLY pointcloud for reference             | `../data/output/arkit_scenes/raw/Training` |
| `rendering.create_mesh`                 | Whether to create and display a mesh                      | `false`                           |
| `interactive_search.outlier_method`     | Outlier detection method (`adaptive`, `iqr`, `percentile`, `z_score`, `combined`) | `"adaptive"`                      |
| `interactive_search.use_statistical_outliers` | Use statistical outlier detection                        | `true`                            |
| `interactive_search.use_dino_filtering` | Use DINO features for structural filtering                | `true`                            |
| `server.port`                           | Port for Rerun viewer or gRPC server                      | `9878`                            |
| `server.mode`                           | `local` (spawn viewer) or `remote` (gRPC server)          | `remote`                          |

**To see all config options and their defaults, check `rerun/config/base_config.yaml`.**

---

#### **Legacy/Direct Script Usage**

You can also run the visualization directly (bypassing Hydra/config):

```bash
python rerun/src/visualize_interactive_text_search.py <path/to/featurized/pt/files> --file-type <file_key>
```
But **using `run_interactive.py` with Hydra is recommended** for full config flexibility.

---

#### **MASt3R-SLAM Results Visualization**

For visualizing raw SLAM outputs (PLY, trajectory, keyframes, depth maps):

```bash
python rerun/scripts/visualize_mast3r_pointcloud.py <slam_dir> [--mode serve|save] [--remote-host <ip>] [--remote-port <port>]
```

---

### Features

- **Interactive text search** on featurized point clouds (CLIP/DINO)
- **Automatic file discovery** from SLAM/featurized output directories
- **3D visualization** of point clouds, camera trajectory, keyframes, depth maps
- **Remote streaming** and local viewer support
- **Configurable outlier detection and filtering** (see terminal `help`)

---

## Development Notes (TUM Infrastructure)

### Server Login
Login (after copying ssh key to server with ssh-copy-id -i ~/.ssh/id_ed25519.pub -o Port=58022 s0125@atcremers45.in.tum.de)

**Server Status Monitor**: [https://adm9.in.tum.de/status](https://adm9.in.tum.de/status)

| Server | SSH Command | RAM | GPU VRAM | Notes |
|--------|-------------|-----|----------|-------|
| atcremers45 | `ssh atcremers45.in.tum.de` | 16 GB | 12 GB | Also available: 45-66, 75, 76 |
| atcremers71 | `ssh atcremers71.in.tum.de` | 64 GB | 16 GB | |
| atcremers72 | `ssh atcremers72.cvai.cit.tum.de` | 32 GB | 16 GB | |
| devcube1 | `ssh devcube1.cvai.cit.tum.de` | 255 GB | 24 GB | High-end server |
| devcube2 | `ssh devcube2.cvai.cit.tum.de` | 255 GB | 24 GB | High-end server |

Copy stuff to server (example)
```bash
rsync -avz -e "ssh -p 58022" /home/marco/Marco/AFM-3D-Search/data/ s0125@atcremers45.in.tum.de:~/AFM-3D-Search/data/
```

Sync data dir (example)
```
rsync -avz -e "ssh -p 58022" s0125@atcremers45.in.tum.de:~/AFM-3D-Search/data /home/marco/Marco/AFM-3D-Search/
rsync -avz -e "ssh -p 58022" /home/marco/Marco/AFM-3D-Search/data s0125@atcremers45.in.tum.de:~/AFM-3D-Search/
```

### Port Forwarding
```
ssh -L 9878:localhost:9878 atcremers45.in.tum.de
ssh -L 9878:localhost:9878 runpod
```

### Useful Commands

#### GPU Commands
```bash
nvidia-smi #Overview
```

#### CPU Usage
```bash
htop
```

#### Disk Space
```bash
df -f
```

#### Debugging Session
```bash
salloc --nodes=1 --cpus-per-task=4 --mem=32G --gres=gpu:1,VRAM:24G --time=0-12:00:00 --mail-type=NONE --part=PRACT --qos=practical_course
```
