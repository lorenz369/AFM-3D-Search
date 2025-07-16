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
| atcremers45 | `ssh -p 58022 s0125@atcremers45.in.tum.de` | 16 GB | 12 GB | Also available: 45-66, 75, 76 |
| atcremers71 | `ssh -p 58022 s0125@atcremers71.in.tum.de` | 64 GB | 16 GB | |
| atcremers72 | `ssh -p 58022 s0125@atcremers72.cvai.cit.tum.de` | 32 GB | 16 GB | |
| devcube1 | `ssh -p 58022 s0125@devcube1.cvai.cit.tum.de` | 255 GB | 24 GB | High-end server |
| devcube2 | `ssh -p 58022 s0125@devcube2.cvai.cit.tum.de` | 255 GB | 24 GB | High-end server |

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
ssh -L 9878:localhost:9878 -p 58022 s0125@devcube1.cvai.cit.tum.de
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

Interactive 3D visualization of MASt3R-SLAM results including PLY point clouds, camera trajectory, keyframes, and depth maps using Rerun SDK.

### Environment Setup

#### Using uv
```bash
# Create and activate the virtual environment
uv venv .rerun_env --python 3.11
source .rerun_env/bin/activate

# Install required packages
uv pip install -r environments/rerun_requirements.txt
```

### Usage

The visualization script automatically discovers and loads all SLAM results from a directory:

```bash
# Local visualization (opens Rerun viewer locally)
python visualize_pointcloud.py <path/to/slam/results/directory>

# Example:
python visualize_pointcloud.py logs/depth_video2_5_sec_test
```

### Script Options

| Option | Description | Default |
|--------|-------------|---------|
| `slam_dir` | Path to SLAM results directory (positional argument) | Required |
| `--mode` | Visualization mode: 'serve' or 'save' | `serve` |
| `--remote-host` | Remote host IP for streaming | None (local) |
| `--remote-port` | Remote port for streaming | `9876` |

#### Additional Usage Examples

```bash
# Stream to remote Rerun viewer
python visualize_pointcloud.py <slam_dir> --remote-host <remote_ip> --remote-port 9876

# Save visualization data to file
python visualize_pointcloud.py <slam_dir> --mode save
```

#### Features

- **Automatic file discovery** from SLAM output directory
- **Interactive 3D visualization** of point clouds with colors
- **Camera trajectory** visualization over time
- **Timeline scrubbing** through keyframe images
- **Synchronized depth maps** with keyframes
- **Remote streaming** support for visualization
- **Timeline-based navigation** using timestamps

#### Auto-discovered Files

The script automatically finds and loads:
- PLY pointcloud file
- Camera poses and timestamps
- Camera intrinsics
- Keyframe images (PNG format)
- Depth maps (NPY format)