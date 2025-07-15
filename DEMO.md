# AFM 3D Search

---

## ⚙️ Setup

```
# 1. Install uv (a fast Python package installer)
curl -LsSf [https://astral.sh/uv/install.sh](https://astral.sh/uv/install.sh) | sh
source $HOME/.local/bin/env

# 2. Create venv and install dependencies
uv venv
source .venv/bin/activate
uv pip install -e .
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