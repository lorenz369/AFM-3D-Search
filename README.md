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
- [MASt3R SLAM](#mast3r-slam)
  - [Environment Setup](#environment-setup)
  - [Checkpoints](#checkpoints)
- [Visualization of Point Clouds](#visualization-of-point-clouds)
  - [Environment Setup](#environment-setup-1)
  - [Usage](#usage)
  - [Script Options](#script-options)
- [Locate-3D Preprocessing Environment Setup](#locate-3d-preprocessing-environment-setup)
  - [Prerequisites](#prerequisites)
  - [Environment Creation and Setup](#environment-creation-and-setup)
  - [Usage](#usage-1)
  - [Output Files](#output-files)
  - [Troubleshooting](#troubleshooting)

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

This repository includes the following open-source submodules:
- `locate-3d`: Facebook's Locate3D library
- `MASt3R-SLAM`: MASt3R-SLAM library

### Server Login
Login (after copying ssh key to server with ssh-copy-id -i ~/.ssh/id_ed25519.pub -o Port=58022 s0125@atcremers45.in.tum.de)

**Server Status Monitor**: [https://adm9.in.tum.de/status](https://adm9.in.tum.de/status)

| Server | SSH Command | RAM | GPU VRAM | Notes |
|--------|-------------|-----|----------|-------|
| atcremers45 | `ssh -p 58022 s0125@atcremers45.in.tum.de` | 16 GB | 12 GB | Also available: 45-66, 75, 76 |
| atcremers71 | `ssh -p 58022 s0125@atcremers71.in.tum.de` | 64 GB | 16 GB | |
| atcremers72 | `ssh -p 58022 s0125@atcremers72.cvai.cit.tum.de` | 32 GB | 16 GB | |
| devcube1 | `ssh -p 58022 s0125@devcube1.cvai.cit.tum.de` | 251 GB | 24 GB | High-end server |
| devcube2 | `ssh -p 58022 s0125@devcube2.cvai.cit.tum.de` | 251 GB | 24 GB | High-end server |

Copy stuff to server (example)
```bash
rsync -avz -e "ssh -p 58022" /home/marco/Marco/AFM-3D-Search/data/ s0125@atcremers45.in.tum.de:~/AFM-3D-Search/data/
rsync -avz -e "ssh -p 58022" /home/marco/Marco/AFM-3D-Search/data/zed s0125@atcremers45.in.tum.de:~/AFM-3D-Search/data/zed
```

Sync MAST3R SLAMS output (example)
```
rsync -avz -e "ssh -p 58022" s0125@atcremers45.in.tum.de:~/AFM-3D-Search/MASt3R-SLAM/logs/ /home/marco/Marco/AFM-3D-Search/MASt3R-SLAM/logs/
```

Sync locate-3d preprocessing output (example)
```
rsync -avz -e "ssh -p 58022" s0125@atcremers45.in.tum.de:~/AFM-3D-Search/locate-3d/preprocessing/output_pointclouds/ /home/marco/Marco/AFM-3D-Search/locate-3d/preprocessing/output_pointclouds/
rsync -avz -e "ssh -p 58022" s0125@atcremers45.in.tum.de:~/AFM-3D-Search/data/zed /home/marco/Marco/AFM-3D-Search/data/zed
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

## MASt3R SLAM
MASt3R-SLAM is included as a submodule in the `MASt3R-SLAM/` directory.

### Environment Setup

#### Using uv
```bash
# Create and activate the virtual environment
uv venv .mast3r-slam_venv --python 3.11
source .mast3r-slam_venv/bin/activate

# Clone the repo (do this before installing local packages)
git clone https://github.com/rmurai0610/MASt3R-SLAM.git --recursive
cd MASt3R-SLAM/
# if you've cloned the repo without --recursive, run this after cd MASt3R-SLAM/:
# git submodule update --init --recursive

# Install PyTorch with matching CUDA version
# Choose one of the following based on your system's CUDA toolkit:

# For CUDA 12.1 (or other CUDA 12.x versions like 12.4):
uv pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121

# Ensure build tools are present in the environment
uv pip install --upgrade setuptools wheel cython

# IMPORTANT for CUDA custom kernel compilation:
module load cuda/12.1.1
# OPTIONAL: Verify cuda installation path
echo $CUDA_HOME
# Load compatible gcc version
module load compiler/gcc-10.1

# Install dependencies
uv pip install --no-build-isolation -e thirdparty/mast3r
uv pip install -e thirdparty/in3d
uv pip install --no-build-isolation -e .

# Optionally install torchcodec for faster mp4 loading
uv pip install torchcodec==0.1
```

#### Using conda
```bash
conda create -n mast3r-slam python=3.11
conda activate mast3r-slam
```

Install pytorch with matching CUDA version:
```bash
# CUDA 11.8
conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1  pytorch-cuda=11.8 -c pytorch -c nvidia
# CUDA 12.1
conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.1 -c pytorch -c nvidia
# CUDA 12.4
conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.4 -c pytorch -c nvidia
```

Clone and install dependencies:
```bash
git clone https://github.com/rmurai0610/MASt3R-SLAM.git --recursive
cd MASt3R-SLAM/

# If you've cloned the repo without --recursive run:
git submodule update --init --recursive

pip install -e thirdparty/mast3r
pip install -e thirdparty/in3d
pip install --no-build-isolation -e .
```

### Checkpoints
Setup the checkpoints for MASt3R and retrieval.
The license for the checkpoints and more information on the datasets used is written here.
```bash
mkdir -p checkpoints/
wget https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth -P checkpoints/
wget https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_trainingfree.pth -P checkpoints/
wget https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_codebook.pkl -P checkpoints/
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
uv pip install -r environments/mast3r_slam_visualization_requirements.txt
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

## Locate-3D Preprocessing Environment Setup

Complete setup guide for the locate-3d preprocessing environment, including system dependencies, Python environment creation, and package installation.

### Prerequisites

First, check that all system-level dependencies are installed:

```bash
# Navigate to environments directory
cd environments

# Check system dependencies
./l3d-check-system-deps.sh
```

If missing packages are reported, install them:
```bash
sudo apt update && sudo apt install -y \
    build-essential cmake pkg-config ffmpeg libopencv-dev libhdf5-dev \
    libjpeg-dev libpng-dev libtiff-dev libwebp-dev libopenjp2-7-dev \
    libavcodec-dev libavformat-dev libswscale-dev libswresample-dev \
    libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev libgtk-3-dev \
    qt6-base-dev libssl-dev libcurl4-openssl-dev zlib1g-dev libbz2-dev \
    liblzma-dev libxml2-dev libxslt1-dev libffi-dev libsqlite3-dev \
    libedit-dev libncurses-dev libreadline-dev tk-dev libgdbm-dev \
    libdb-dev libpcap-dev xz-utils curl llvm libgdbm-compat-dev libc6-dev
```

### Environment Creation and Setup

#### Step 1: Create Virtual Environment
```bash
# Create Python 3.11 environment for locate-3d preprocessing
uv venv .l3d_preprocessing --python 3.11
source .l3d_preprocessing/bin/activate
```

#### Step 2: Install PyTorch First
Install PyTorch with CUDA support before other packages to avoid build issues:
```bash
# For CUDA 12.x (adjust version as needed)
uv pip install torch>=2.7.0 torchvision>=0.22.0
```

#### Step 3: Install Core Requirements
```bash
# Install the main requirements (with PyTorch geometric packages commented out)
uv pip install -r environments/l3d_preprocessing.txt
```

#### Step 4: Install PyTorch Geometric
Install PyTorch Geometric packages separately with the correct CUDA version:
```bash
# Check your PyTorch version first
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.version.cuda}')"

# Install PyTorch Geometric (adjust CUDA version to match your setup)
# For PyTorch 2.7.x with CUDA 12.6:
uv pip install torch-geometric torch-cluster torch-scatter torch-sparse torch-spline-conv \
    --find-links https://data.pyg.org/whl/torch-2.7.0+cu126.html

# For other versions, check: https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html
```

### Usage

#### Processing MASt3R-SLAM Output
```bash
# Navigate to preprocessing directory
cd locate-3d/preprocessing

# Process MASt3R-SLAM data with both CLIP and DINO features
python mast3r_slam_wrapper.py \
    --mast3r_data_dir ../../MASt3R-SLAM/logs/your_scene \
    --poses_file ../../MASt3R-SLAM/logs/your_scene/scene_with_intrinsics.txt \
    --output_dir output_pointclouds \
    --scene_name your_scene_name \
    --config_type both
```

#### Available Feature Types
- `--config_type clip`: Extract only CLIP features
- `--config_type dino`: Extract only DINO features  
- `--config_type both`: Extract both CLIP and DINO features (recommended)

### Output Files

The preprocessing generates featurized point clouds:
- `scene_name_clip.pt`: Point cloud with CLIP features
- `scene_name_dino.pt`: Point cloud with DINO features
- `scene_name_combined.pt`: Point cloud with both feature types

### Troubleshooting

#### Common Issues

**PyTorch Geometric Build Failures:**
- Solution: Install PyTorch first, then install geometric packages separately with `--find-links`

**CUDA Version Mismatch:**
- Check your CUDA version: `nvidia-smi`
- Use matching PyTorch CUDA version in installation commands
- The requirements use `>=12.6.0` for CUDA packages to support CUDA 12.6+

**Missing System Dependencies:**
- Run `./l3d-check-system-deps.sh` to identify missing packages
- Install missing packages with the provided `apt install` command

**Memory Issues:**
- Use `--max_frames` parameter to limit processing for testing
- Check available RAM: `free -h`
