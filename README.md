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
rsync -avz -e "ssh -p 58022" /Users/marcolorenz/Library/CloudStorage/OneDrive-Personal/*.MOV s0125@atcremers45.in.tum.de:~/
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

Interactive 3D visualization of PLY point clouds using Rerun SDK, with support for synchronized keyframe images.

### Environment Setup

#### Using uv
```bash
# Create and activate the virtual environment
uv venv .rerun_env
source .rerun_env/bin/activate

# Install required packages
uv pip install rerun-sdk plyfile pillow
```

### Usage

The visualization script supports PLY point clouds with optional keyframe images:

```bash
# Basic usage with default paths
python visualize_pointcloud.py

# Visualize only the pointcloud (skip keyframes)
python visualize_pointcloud.py --no-keyframes

# Specify custom paths
python visualize_pointcloud.py --ply path/to/your.ply --keyframes path/to/keyframes/
```

## Segment frames

Segment all video frames with SAM.

### Environment Setup

#### Using uv
```bash
# Create and activate the virtual environment
uv venv .segment_venv --python 3.11
source .segment_venv/bin/activate

# Install required packages
uv pip install torch torchvision torchaudio
uv pip install git+https://github.com/facebookresearch/segment-anything.git

wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth -O sam_vit_h.pth

```
### Script Options

| Option | Description | Default |
|--------|-------------|---------|
| `--ply` | Path to PLY pointcloud file | `data/mast3r_results/AFM_Video_Marco_1.ply` |
| `--keyframes` | Path to keyframes directory | `data/mast3r_results/keyframes/AFM_Video_Marco_1` |
| `--no-keyframes` | Skip loading keyframe images | False |

#### Features

- **Interactive 3D visualization** of point clouds with colors
- **Timeline scrubbing** through keyframe images
- **Synchronized display** of 3D reconstruction and original camera views
- **Automatic color normalization** for PLY files
- **Error handling** for missing files or corrupted images

#### Requirements

- PLY files with vertex coordinates (x, y, z)
- Optional: RGB color data (red, green, blue fields)
- Optional: Keyframe images in PNG format with timestamp filenames
