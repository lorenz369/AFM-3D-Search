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
- [DUSt3R](#dust3r)
  - [Environment Setup](#environment-setup)
  - [Demo](#demo)
- [MASt3R SLAM](#mast3r-slam)
  - [Environment Setup](#environment-setup-1)
  - [Checkpoints](#checkpoints)
- [Visualization of Point Clouds](#visualization-of-point-clouds)
  - [Environment Setup](#environment-setup-2)
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

This repository includes the following submodules:
- `locate-3d`: Facebook's Locate3D library
- `dust3r`: NAVER LABS Europe's DUSt3R library
- `MASt3R-SLAM`: MASt3R-SLAM library

### Server Login
Login (after copying ssh key to server with ssh-copy-id -i ~/.ssh/id_ed25519.pub -o Port=58022 s0125@atcremers45.in.tum.de)
```bash
ssh -p 58022 s0125@atcremers45.in.tum.de
```

Switch node (with pw)
```bash
ssh s0125@atcremers46.in.tum.de
```

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

## DUSt3R
DUSt3R is included as a submodule in the `dust3r/` directory.

### Environment Setup

#### Using uv
To install using the recommended script, run:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
```

Dependencies with uv:
```bash
git clone --recursive https://github.com/naver/dust3r
uv venv .dust3r_venv --python 3.11
source .dust3r_venv/bin/activate
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
cd dust3r
uv pip install -r requirements.txt
# DUST3R relies on RoPE positional embeddings for which you can compile some cuda kernels for faster runtime.
cd croco/models/curope/
python setup.py build_ext --inplace
cd ../../../
```

#### Using conda
```bash
conda create -n dust3r python=3.11 cmake=3.14.0
conda activate dust3r 
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia  # use the correct version of cuda for your system
pip install -r requirements.txt
# Optional: you can also install additional packages to:
# - add support for HEIC images
# - add pyrender, used to render depthmap in some datasets preprocessing
# - add required packages for visloc.py
pip install -r requirements_optional.txt
```

Additional setup:
```bash
# DUST3R relies on RoPE positional embeddings for which you can compile some cuda kernels for faster runtime.
cd croco/models/curope/
python setup.py build_ext --inplace
cd ../../../
```

### Demo
```bash
python3 demo.py --model_name DUSt3R_ViTLarge_BaseDecoder_224_linear
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
