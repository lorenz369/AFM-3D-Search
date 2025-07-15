# AFM 3D Search

---

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

## 🚀 Usage

```
# 1. Generate a feature cloud from your images
# Note: Place images in the default path 'data/input/bude/images'
# or edit the path in 'src/afm_3d_search/conf/paths/default.yaml'
python src/afm_3d_search/generate_pointcloud.py

# 2. Run analysis scripts on the latest generated cloud

# Highlight points using a text query (CLIP)
python src/afm_3d_search/highlight_clip.py text_query="a red chair"

# Highlight points by similarity to a point index (DINO)
python src/afm_3d_search/highlight_dino.py query_point_index=100000 top_k=5000
```

## New Usage
There are two primary ways to run this project: directly from the command line for local experiments, or as a full server-worker pipeline to handle API requests.

### 1. Local Experiments & Testing
Use this method to run the entire pipeline on local data. This is ideal for development, debugging, and experimenting with different model configurations. The default test data is expected in data/testing/bude/images.

#### Run the pipeline with default settings:

```
# This will process the default test data using the models defined in your config
python src/afm_3d_search/run_pipeline.py
```

#### Run with different model configurations:
You can easily override any setting from the command line using Hydra's syntax.

```
# Run with faster, lower-quality models for a quick test
python src/afm_3d_search/run_pipeline.py models.dino=small models.clip=base models.sam=base
```

### 2. Full Server Pipeline (via API)
Use this method to run the live service that accepts image uploads. This requires two separate terminals.

#### Terminal 1: Start the API Server
This command starts the web server, which listens for uploads.


```
uvicorn src.afm_3d_search.api.main:app --host 0.0.0.0 --port 8000 --reload
```

#### Terminal 2: Start the Worker
This command starts the background worker, which will process jobs as they are created by the API server.

```
python src/afm_3d_search/worker.py
Once both are running, you can send a POST request with multiple image files to the endpoint http://localhost:8000/v1/scenes. The server will accept the upload and the worker will begin processing the scene.
```