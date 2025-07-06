# Environment Setup Instructions

This guide shows how to recreate your conda environment using system packages + uv.

## Prerequisites

1. Install Python 3.10 (use pyenv, system package manager, or download from python.org)
2. Install uv: `curl -LsSf https://astral.sh/uv/install.sh | sh`

## Step 1: Install System Dependencies

First, install the system-level dependencies that conda was managing:

```bash
# See system-dependencies.md for the full list
# For Ubuntu/Debian:
sudo apt update
sudo apt install -y build-essential cmake pkg-config ffmpeg libopencv-dev libhdf5-dev ...
```

## Step 2: Create Python Environment with uv

```bash
# Create a new Python environment
uv venv locate-3d-preprocessing --python 3.10

# Activate the environment
source locate-3d-preprocessing/bin/activate
# or on Windows: locate-3d-preprocessing\Scripts\activate
```

## Step 3: Install Python Packages

```bash
# Install all Python packages with uv
uv pip install -r requirements.txt
```

## Alternative: One-command setup with uv

You can also let uv handle the environment creation and package installation in one go:

```bash
# Create project and install dependencies
uv add --requirement requirements.txt
```

## Important Notes

### OpenCV Considerations
Your conda environment used the conda-forge opencv package. You have two options:

1. **Use system OpenCV** (recommended): The system packages include OpenCV, and Python bindings should work
2. **Use pip OpenCV**: Uncomment the `opencv-python` line in requirements.txt

### CUDA Support
The CUDA packages are included in requirements.txt, but you need:
1. NVIDIA drivers installed
2. CUDA toolkit installed (if not using the pip CUDA packages)

### Version Matching
Some packages might have slightly different versions available on PyPI vs conda-forge. The requirements.txt uses the exact versions from your conda environment, but you might need to adjust if some aren't available.

## Verification

Test that everything works:

```bash
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
python -c "import cv2; print('OpenCV version:', cv2.__version__)"
python -c "import numpy; print('NumPy version:', numpy.__version__)"
```

## Troubleshooting

### Package not found
If a package isn't available with the exact version:
```bash
# Remove version constraint and let uv find compatible version
uv pip install package_name
```

### System dependency missing
If you get import errors, check that system dependencies are installed:
```bash
pkg-config --list-all | grep opencv  # Check OpenCV
ldconfig -p | grep cuda             # Check CUDA libraries
```

### Environment conflicts
If you run into issues, start fresh:
```bash
rm -rf locate-3d-preprocessing
uv venv locate-3d-preprocessing --python 3.10
source locate-3d-preprocessing/bin/activate
uv pip install -r requirements.txt
``` 