# AFM 3D Search


## ⚙️  Setup

```
# 1. Install uv (a fast Python package installer)
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

# 2. Clone Repo
git clone https://github.com/lorenz369/AFM-3D-Search.git
git switch feat/add-scene-processing-endpoint
cd AFM-3D-Search/
git submodule update --init --recursive

# 3. Create venv and install dependencies
uv venv
source .venv/bin/activate
uv pip install -e .

# 4. install vggt
cd submodules/vggt
uv pip install -e .
cd ../..
```

setup.sh

````
#!/bin/bash

set -e  # Exit on any error

echo "🚀 Starting AFM-3D-Search setup..."

# 1. Install uv (a fast Python package installer)
echo "📦 Installing uv..."
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

# 2. Clone Repo
echo "📥 Cloning repository..."
git clone https://github.com/lorenz369/AFM-3D-Search.git
cd AFM-3D-Search/
git switch feat/add-scene-processing-endpoint
git submodule update --init --recursive

# 3. Create venv and install dependencies
echo "🐍 Creating virtual environment and installing dependencies..."
uv venv
source .venv/bin/activate
uv pip install -e .

# 4. Install vggt
echo "🔧 Installing vggt submodule..."
cd submodules/vggt
uv pip install -e .
cd ../..

echo "✅ Setup complete! Virtual environment is activated."
echo "💡 To activate the environment in the future, run: source AFM-3D-Search/.venv/bin/activate"
````

```
chmod +x setup.sh
./setup.sh
```

## Usage

There are three primary ways to use this project: running the full pipeline on a local dataset, analyzing the results of a processed scene, or deploying the full API service.

A **`scene_id`** is required for most commands and corresponds to the name of a scene's data folder (e.g., `bude`).

### 1. Running the Full Pipeline

Use this method to process a directory of images into a 3D point cloud with extracted features. This is ideal for development and testing.

The pipeline reads images from `data/testing/<scene_id>/images/` and saves the final artifacts to `data/completed/<scene_id>/`.

#### **Standard Run**

```bash
# Process the 'bude' scene with the default models
python src/afm_3d_search/run_pipeline.py scene_id=bude
```

#### **Run with Different Models**

You can easily swap model configurations by referencing other files in the `conf/models/` directory. This is perfect for quick experiments.

```bash
# Run with faster, lower-VRAM models defined in 'conf/models/fast_test.yaml'
python src/afm_3d_search/run_pipeline.py scene_id=bude models=fast_test
```

### 2. Analyzing a Processed Scene

Once a scene has been processed, you can use the highlight scripts to perform analysis. These scripts read from `data/completed/<scene_id>/` and save their output (a colored `.ply` file) to a new, timestamped folder in `outputs/`.

#### **Text-based Search (CLIP)**

```bash
# Analyze the 'bude' scene with the default text query
python scripts/highlight_clip.py scene_id=bude

# Override the text query from the command line
python scripts/highlight_clip.py scene_id=bude highlight.text_query="a red chair"
```

#### **Similarity Search (DINO)**

```bash
# Analyze 'bude' using a default starting point for similarity search
python scripts/highlight_dino.py scene_id=bude

# Specify a different point index and search radius (top_k)
python scripts/highlight_dino.py scene_id=bude highlight.query_point_index=50000 highlight.top_k=1000
```

### 3. Running as an API Service

This runs the project as a live service that accepts image uploads via an HTTP endpoint. This requires two separate terminals.

#### **Terminal 1: Start the API Server**

This command starts the web server, which listens for uploads on port 8000. Uploaded images are saved to the `data/staging/` directory.

```bash
uvicorn src.afm_3d_search.api.main:app --host 0.0.0.0 --port 8000 --reload
```

#### **Terminal 2: Start the Worker**

This command starts the background worker, which polls for new jobs created by the API server and processes them using the main pipeline.

```bash
python src/afm_3d_search/worker.py
```
Once both services are running, you can `POST` multiple image files to the `http://localhost:8000/v1/scenes` endpoint to create a new processing job.