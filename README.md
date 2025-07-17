# AFM Group 2

## Data
Large datasets and results are available in our [Google Drive folder](https://drive.google.com/drive/folders/184vJEGNb4RQ5tb9fF1LaFxy98oRriyPi?usp=drive_link).

## Table of Contents
- [Setup](#setup)
  - [Git Submodules](#git-submodules)
  - [Server Login](#server-login)
- [Useful Commands](#useful-commands)
  - [GPU Commands](#gpu-commands)
  - [CPU Usage](#cpu-usage)
  - [Disk Space](#disk-space)
  - [Debugging Session](#debugging-session)
- [Visualization of Point Clouds](#visualization-of-point-clouds)
  - [Environment Setup](#environment-setup-1)
  - [Usage](#usage)
  - [Script Options](#script-options)

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

## Port Forwarding
```
ssh -L 9878:localhost:9878 atcremers45.in.tum.de
ssh -L 9878:localhost:9878 runpod
```

## Useful Commands

### GPU Commands
```bash
nvidia-smi #Overview
```

### CPU Usage
```bash
htop
```

### Disk Space
```bash
df -f
```

### Debugging Session
```bash
salloc --nodes=1 --cpus-per-task=4 --mem=32G --gres=gpu:1,VRAM:24G --time=0-12:00:00 --mail-type=NONE --part=PRACT --qos=practical_course
```

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