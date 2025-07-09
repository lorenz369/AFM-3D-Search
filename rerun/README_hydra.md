# Featurized Pointcloud Visualization with Hydra Configuration

A powerful, configuration-driven interface for visualizing featurized pointclouds with CLIP and DINO features, supporting text-based semantic search, surface reconstruction, and interactive highlighting.

## Quick Start

```bash
# Basic visualization
python visualize_featurized_pointcloud_hydra.py pointcloud_dir=/path/to/data

# Text-based semantic search
python visualize_featurized_pointcloud_hydra.py --config-name=text_search pointcloud_dir=/path/to/data text_similarity.query='red car'

# High-quality with mesh reconstruction
python visualize_featurized_pointcloud_hydra.py pointcloud_dir=/path/to/data rendering.create_mesh=true rendering.point_size=0.02
```

## Configuration Overview

### Available Presets
- **`base_config`** (default): Balanced settings for general exploration
- **`text_search`**: Optimized for CLIP-based semantic search

### Core Configuration Structure
```yaml
pointcloud_dir: ???          # Required: path to .pt files
file_type: "combined"        # File to load: combined, clip, dino
mode: "serve"                # Output mode: serve, web, save

server:
  port: 9878                # Server port

rendering:
  point_size: 0.01          # Point radius
  create_mesh: false        # Enable surface reconstruction
  mesh_method: "poisson"    # Reconstruction algorithm
  use_voxels: false         # Voxel downsampling
  voxel_size: 0.05         # Voxel grid size

features:
  show_features: true       # Show feature visualizations
  feature_type: "both"      # clip, dino, both
  pca_method: "hsv"        # Feature-to-color mapping

highlighting:
  enable: false            # Enable point highlighting
  mode: "random_points"    # Highlighting strategy
  animate: false           # Dynamic highlighting
  animation_duration: 30.0 # Animation time (seconds)

text_similarity:
  query: null              # Text search query
  top_k: 100              # Number of results
  threshold: 0.3          # Similarity threshold
  clip_model_version: "ViT-B/32"  # CLIP model
```

## Complete Configuration Reference

### Root Level Settings

#### `pointcloud_dir` (Required)
- **Type**: String (path)
- **Default**: None (must be provided)
- **Description**: Absolute or relative path to directory containing featurized pointcloud `.pt` files
- **Example**: `/home/user/data/pointclouds` or `./processed_data`
- **Notes**: Script auto-discovers all `.pt` files in this directory

#### `file_type`
- **Type**: String
- **Default**: `"combined"`
- **Options**: 
  - `"combined"`: Use combined features file (both CLIP and DINO)
  - `"clip"`: Use CLIP features only (best for text search)
  - `"dino"`: Use DINO features only (dense visual features)
  - Custom filename without `.pt` extension
- **Description**: Selects which featurized pointcloud file to load from the discovered files
- **When to use**: 
  - `"combined"` for general exploration
  - `"clip"` for semantic search applications
  - `"dino"` for detailed visual feature analysis

#### `mode`
- **Type**: String
- **Default**: `"serve"`
- **Options**:
  - `"serve"`: Start gRPC server for Rerun viewer connection
  - `"web"`: Launch web interface (auto-opens browser on port 9090)
  - `"save"`: Save visualization to .rrd file without viewer
- **Description**: Determines how the visualization is output and accessed
- **When to use**:
  - `"serve"` for production, remote access, or when using Rerun desktop app
  - `"web"` for demos, presentations, or quick exploration
  - `"save"` for batch processing or offline analysis

### Server Configuration

#### `server.port`
- **Type**: Integer
- **Default**: `9878`
- **Range**: `1024-65535` (recommended: `9000-9999`)
- **Description**: Port number for gRPC server when using `mode: "serve"`
- **Notes**: Only affects `"serve"` mode; web mode always uses port 9090
- **When to change**: If default port is already in use or for specific network configurations

### Rendering Configuration

#### `rendering.point_size`
- **Type**: Float
- **Default**: `0.01`
- **Range**: `0.001-0.1`
- **Units**: 3D world units (typically meters)
- **Description**: Radius of rendered points in the 3D scene
- **Guidelines**:
  - `0.005-0.01`: Dense pointclouds (>500K points), detailed analysis
  - `0.01-0.02`: Standard exploration, balanced visibility
  - `0.02-0.05`: Sparse pointclouds, presentations, emphasis
  - `>0.05`: Very sparse data or stylistic effects

#### `rendering.create_mesh`
- **Type**: Boolean
- **Default**: `false`
- **Description**: Whether to generate surface mesh from pointcloud using surface reconstruction
- **Performance impact**: Significant processing time for large pointclouds
- **When to enable**: 
  - Continuous surface analysis needed
  - Publication-quality visualizations
  - Understanding object topology
- **When to disable**: 
  - Quick exploration
  - Large datasets (>1M points)
  - Point-based analysis sufficient

#### `rendering.mesh_method`
- **Type**: String
- **Default**: `"poisson"`
- **Options**: `"poisson"`, `"ball_pivoting"`, `"alpha_shape"`
- **Description**: Algorithm for surface reconstruction (only used if `create_mesh: true`)
- **Algorithm details**: See "Surface Reconstruction Theory" section below

#### `rendering.use_voxels`
- **Type**: Boolean
- **Default**: `false`
- **Description**: Whether to downsample pointcloud using voxel grid averaging
- **Purpose**: Reduces point density for performance while preserving overall structure
- **When to enable**:
  - Large pointclouds (>1M points)
  - Performance optimization needed
  - Uniform point density desired
- **Trade-offs**: Faster rendering vs. loss of fine detail

#### `rendering.voxel_size`
- **Type**: Float
- **Default**: `0.05`
- **Range**: `0.01-0.2`
- **Units**: 3D world units (typically meters)
- **Description**: Size of voxel grid cells for downsampling (only used if `use_voxels: true`)
- **Guidelines**:
  - `0.01-0.03`: Minimal detail loss, moderate performance gain
  - `0.05`: Balanced performance (default)
  - `0.07-0.1`: Aggressive downsampling, maximum performance
  - `>0.1`: Very coarse representation, only for overview

### Feature Visualization Configuration

#### `features.show_features`
- **Type**: Boolean
- **Default**: `true`
- **Description**: Whether to create feature-based color visualizations using PCA
- **When to disable**: 
  - Only want RGB pointcloud visualization
  - Performance optimization
  - Presentation mode without technical overlays

#### `features.feature_type`
- **Type**: String
- **Default**: `"both"`
- **Options**:
  - `"both"`: Show both CLIP and DINO feature visualizations
  - `"clip"`: Show only CLIP features (semantic, text-searchable)
  - `"dino"`: Show only DINO features (dense visual patterns)
- **Description**: Which types of learned features to visualize
- **Feature characteristics**:
  - CLIP: Semantic understanding, good for object recognition, text alignment
  - DINO: Dense visual patterns, good for texture and geometric details

#### `features.pca_method`
- **Type**: String
- **Default**: `"hsv"`
- **Options**: `"hsv"`, `"rgb"`
- **Description**: Method for mapping high-dimensional features to RGB colors
- **Technical details**:
  - `"hsv"`: Maps first PCA component to hue, second to saturation, third to value
  - `"rgb"`: Direct mapping of first three PCA components to red, green, blue channels
- **Visual differences**: HSV typically provides better perceptual color separation

### Highlighting Configuration

#### `highlighting.enable`
- **Type**: Boolean
- **Default**: `false`
- **Description**: Whether to add highlighting overlays to emphasize specific points
- **Purpose**: Draw attention to regions of interest, create interactive exploration
- **Performance impact**: Minimal

#### `highlighting.mode`
- **Type**: String
- **Default**: `"random_points"`
- **Options**:
  - `"random_points"`: Randomly selected points across the pointcloud
  - `"spatial_clusters"`: Spatially coherent regions (good for showing structure)
  - `"tight_clusters"`: Small, focused clusters for detailed examination
  - `"feature_based"`: Points with extreme feature values (outliers/interesting features)
  - `"voxel_highlights"`: Grid-based highlighting with voxel centers
- **Description**: Strategy for selecting which points to highlight

#### `highlighting.animate`
- **Type**: Boolean
- **Default**: `false`
- **Description**: Whether highlights change over time to create dynamic visualization
- **Use cases**: Presentations, time-based exploration, drawing attention
- **Timeline**: Creates Rerun timeline that can be scrubbed through

#### `highlighting.animation_duration`
- **Type**: Float
- **Default**: `30.0`
- **Range**: `5.0-600.0`
- **Units**: Seconds
- **Description**: Total duration of highlighting animation cycle (only used if `animate: true`)
- **Guidelines**:
  - `5-15s`: Quick demos, testing
  - `30-60s`: Standard presentations
  - `60-300s`: Background visualization, long-form analysis

### Text-Based Similarity Search Configuration

#### `text_similarity.query`
- **Type**: String or null
- **Default**: `null`
- **Description**: Natural language text query for semantic search in CLIP feature space
- **Examples**: `"red chair"`, `"wooden table"`, `"car door"`, `"person walking"`
- **Requirements**: Only works with CLIP features present in pointcloud
- **Language**: English (CLIP model limitation)

#### `text_similarity.top_k`
- **Type**: Integer
- **Default**: `100`
- **Range**: `1-10000`
- **Description**: Maximum number of most similar points to highlight
- **Guidelines**:
  - `10-50`: Very specific object search
  - `100-300`: Standard semantic categories
  - `500-1000`: Broad concept exploration
  - `>1000`: May overwhelm visualization

#### `text_similarity.threshold`
- **Type**: Float
- **Default**: `0.3`
- **Range**: `0.0-1.0`
- **Description**: Minimum cosine similarity between text query and point features
- **Technical details**: Cosine similarity ranges from -1 (opposite) to 1 (identical)
- **Guidelines**:
  - `0.1-0.2`: Very broad similarity, many results
  - `0.3`: Balanced precision and recall (default)
  - `0.4-0.5`: High precision, exact concept matches
  - `>0.5`: Very strict, may yield few results

#### `text_similarity.clip_model_version`
- **Type**: String
- **Default**: `"ViT-B/32"`
- **Options**: `"ViT-B/32"`, `"ViT-L/14"`
- **Description**: CLIP model architecture used for text encoding
- **Compatibility**: Must match the CLIP model used during pointcloud preprocessing
- **Characteristics**:
  - `"ViT-B/32"`: 512-dimensional features, faster, good performance
  - `"ViT-L/14"`: 768-dimensional features, slower, higher accuracy
- **Auto-detection**: Script attempts to detect correct model from feature dimensions

## Detailed Configuration Reference

### Core Settings

#### `pointcloud_dir` (Required)
Path to directory containing `.pt` pointcloud files from the preprocessing pipeline.

#### `file_type`
- `"combined"`: All features available (default)
- `"clip"`: CLIP features only (best for text search)
- `"dino"`: DINO features only (dense visual features)
- Custom filename (without .pt extension)

#### `mode`
- `"serve"`: gRPC server (most flexible, supports remote connections)
- `"web"`: Web browser interface (auto-opens, good for demos)
- `"save"`: Save to .rrd file (batch processing)

### Rendering Configuration

#### Point Visualization
- `point_size`: Radius of rendered points
  - `0.005-0.01`: Dense pointclouds, detailed analysis
  - `0.015-0.025`: Standard exploration
  - `0.03-0.05`: Sparse pointclouds, presentations

#### Surface Reconstruction Theory

Surface reconstruction creates continuous meshes from discrete point samples. The algorithm choice depends on your data characteristics:

**Poisson Reconstruction** (`mesh_method: "poisson"`)
- **Theory**: Solves a 3D Poisson equation using oriented points (positions + normals). Treats surface reconstruction as finding the function whose gradient best matches the normal field.
- **Strengths**: Robust to noise, fills holes, creates watertight meshes
- **Best for**: Complete objects, organic shapes, noisy data
- **Limitations**: May over-smooth sharp features

**Ball Pivoting** (`mesh_method: "ball_pivoting"`)
- **Theory**: Simulates rolling a ball of radius R over the point set. Creates triangles when the ball touches exactly three points without containing others.
- **Strengths**: Preserves sharp features, fast, respects local point density
- **Best for**: Detailed surfaces, architectural scenes, clean data
- **Limitations**: Sensitive to noise, may create holes in sparse regions

**Alpha Shapes** (`mesh_method: "alpha_shape"`)
- **Theory**: Generalizes convex hull concept using spheres of radius α. Creates simplices (triangles) that can be "probed" by an α-sphere.
- **Strengths**: Captures sharp edges and concavities, parameter-controlled detail
- **Best for**: Sharp geometric features, non-convex shapes
- **Limitations**: Requires parameter tuning, may create disconnected components

#### Voxel Downsampling
- `use_voxels: true`: Reduces point density by averaging points within voxel cells
- `voxel_size`: Cell size for downsampling
  - `0.01-0.03`: High detail retention
  - `0.05`: Balanced performance (default)
  - `0.07-0.1`: Maximum performance

### Feature Visualization

#### `feature_type`
- `"both"`: Show CLIP and DINO feature visualizations
- `"clip"`: CLIP features only (semantic, good for text search)
- `"dino"`: DINO features only (dense visual patterns)

#### `pca_method`
- `"hsv"`: Maps PCA components to hue/saturation/value (better visual separation)
- `"rgb"`: Direct mapping to red/green/blue channels

**PCA Color Mapping Theory**: High-dimensional features (512D CLIP, 1024D DINO) are reduced to 3D via Principal Component Analysis, then mapped to colors. HSV mapping uses the first two components for hue and saturation, providing better perceptual color separation than direct RGB mapping.

### Text-Based Semantic Search

#### `text_similarity.query`
Natural language description of objects to find (e.g., "red chair", "car door")

#### `text_similarity.threshold`
Cosine similarity threshold between text and visual features:
- `0.1-0.2`: Broad similarity, more results
- `0.3`: Balanced precision (default)
- `0.4-0.5`: High precision, exact matches

#### CLIP Model Versions
- `"ViT-B/32"`: 512D features, faster (default)
- `"ViT-L/14"`: 768D features, higher accuracy

**Semantic Search Theory**: CLIP embeds both images and text into a shared vector space where semantically similar content has high cosine similarity. The search finds pointcloud regions whose visual features are closest to the text query embedding.

### Highlighting and Animation

#### `highlighting.mode`
- `"random_points"`: Random point selection
- `"spatial_clusters"`: Spatially coherent regions
- `"tight_clusters"`: Small, focused clusters
- `"feature_based"`: Points with extreme feature values
- `"voxel_highlights"`: Grid-based highlighting

#### `highlighting.animate`
Creates time-varying highlights for dynamic visualization.

## Usage Scenarios

### Initial Data Exploration
```bash
python visualize_featurized_pointcloud_hydra.py \
  pointcloud_dir=/path/to/data \
  mode=web \
  rendering.point_size=0.02 \
  features.feature_type=both
```

### Semantic Object Search
```bash
python visualize_featurized_pointcloud_hydra.py \
  --config-name=text_search \
  pointcloud_dir=/path/to/data \
  text_similarity.query='wooden table' \
  text_similarity.threshold=0.2 \
  text_similarity.top_k=200
```

### High-Quality Analysis with Surface Reconstruction
```bash
python visualize_featurized_pointcloud_hydra.py \
  pointcloud_dir=/path/to/data \
  rendering.create_mesh=true \
  rendering.mesh_method=poisson \
  rendering.point_size=0.015 \
  rendering.use_voxels=true \
  rendering.voxel_size=0.03
```

### Dynamic Presentation
```bash
python visualize_featurized_pointcloud_hydra.py \
  pointcloud_dir=/path/to/data \
  mode=web \
  highlighting.enable=true \
  highlighting.animate=true \
  highlighting.mode=spatial_clusters \
  highlighting.animation_duration=60.0
```

### Performance Optimization for Large Datasets
```bash
python visualize_featurized_pointcloud_hydra.py \
  pointcloud_dir=/path/to/data \
  rendering.use_voxels=true \
  rendering.voxel_size=0.08 \
  features.feature_type=clip \
  highlighting.enable=false
```

## Advanced Configuration

### Custom Configuration Files
Create `my_config.yaml`:
```yaml
defaults:
  - base_config
  - _self_

pointcloud_dir: /path/to/data

rendering:
  create_mesh: true
  mesh_method: ball_pivoting
  point_size: 0.025

text_similarity:
  query: "red car"
  threshold: 0.15
```

Use with: `python visualize_featurized_pointcloud_hydra.py --config-name=my_config`

### Parameter Sweeps
```bash
# Test multiple configurations
python visualize_featurized_pointcloud_hydra.py \
  --multirun \
  pointcloud_dir=/path/to/data \
  rendering.point_size=0.01,0.02,0.03 \
  text_similarity.threshold=0.1,0.2,0.3
```

### Override Patterns
```bash
# Complex overrides
python visualize_featurized_pointcloud_hydra.py \
  --config-name=text_search \
  pointcloud_dir=/data \
  text_similarity.query='blue chair' \
  text_similarity.top_k=500 \
  rendering.create_mesh=true \
  rendering.mesh_method=ball_pivoting \
  highlighting.mode=tight_clusters
```

## Configuration Groups

The configuration system supports modular composition:

```
config/
├── base_config.yaml          # Main configuration
├── text_search.yaml          # Text search preset
├── visualization/            # Rendering presets
├── features/                 # Feature settings
└── highlighting/             # Highlighting modes
```

## Help and Debugging

```bash
# Show all configuration options
python visualize_featurized_pointcloud_hydra.py --help

# Show specific preset
python visualize_featurized_pointcloud_hydra.py --config-name=text_search --help

# Hydra-specific help
python visualize_featurized_pointcloud_hydra.py --hydra-help

# Print final configuration without running
python visualize_featurized_pointcloud_hydra.py --cfg job pointcloud_dir=/path/to/data
```

## Performance Considerations

**For Large Pointclouds (>1M points)**:
- Enable voxel downsampling: `rendering.use_voxels=true`
- Increase voxel size: `rendering.voxel_size=0.06`
- Disable mesh reconstruction: `rendering.create_mesh=false`
- Use single feature type: `features.feature_type=clip`

**For High-Quality Visualization**:
- Enable mesh reconstruction: `rendering.create_mesh=true`
- Use Poisson reconstruction: `rendering.mesh_method=poisson`
- Smaller voxels if downsampling: `rendering.voxel_size=0.02`
- Larger points: `rendering.point_size=0.02`

**For Text Search Optimization**:
- Use CLIP features only: `file_type=clip`, `features.feature_type=clip`
- Lower similarity threshold: `text_similarity.threshold=0.2`
- Enable highlighting: `highlighting.enable=true`

This configuration system provides complete control over the visualization pipeline while maintaining simplicity for common use cases. 