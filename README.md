# retico-sceneGraph
<img width="1797" height="710" alt="image" src="https://github.com/user-attachments/assets/9b481f69-f9f1-41b8-87c9-8473cadf5416" />

A Retico extension for real-time scene graph generation, embedding, memory management, and visualization using the RelTR model.

## Features
- Generate scene graphs from images, extracting objects and their relationships.
- Embed triplets for Retrieval-Augmented Generation (RAG) queries.
- Store and query scene graphs in a memory module across multiple camera sources.
- Visualize scene graphs on images with bounding boxes and relationship arrows.

To tag different camera sources, use the `meta_data` field when creating the video producing module and populate `camera_name` with the camera identifier. This will allow the scene graph module to differentiate between inputs from different cameras.
The methods that allow querying the `SceneGraphMemory` are the following:
- `get_camera_names()`: returns all the cameras available by their name.
- `get_scene_graph(camera_name)`: returns the scene graph for a given camera name.
- `query_camera(camera_name, query)`: queries the memory for a given camera name and returns the scene sub-graph that matches the query and its neighbors.
- `query_memory(query)`: queries the memory for a given query string and returns for each camera the scene sub-graph that matches the query and its neighbors.

## Prerequisites
- Python 3.8 or higher
- PyTorch (with CUDA support for GPU acceleration)
- torchvision compatible with the RelTR codebase
- Transformers and SentenceTransformers
- retico-core and retico-vision libraries
- RelTR repository clone and pretrained checkpoint

## Installation
1. Clone and install this repository:
   ```
   pip install git+https://github.com/retico-team/retico-sceneGraph.git
   ```
2. Clone the [RelTR](https://github.com/yrcong/RelTR.git) repository and set `RelTR_PATH` to the path of the cloned repository.
3. Add `retico-core` and `retico-vision` to your Python path.

## Download Pretrained Model
Download the RelTR checkpoint and place it in the project root:
```bash
gdown "https://drive.google.com/uc?id=1id6oD_iwiNDD6HyCn2ORgRTIKkPD3tUD"
```

## Usage
To see and example of how to use the scene graph unit, refer to the `example.py` runner script.
