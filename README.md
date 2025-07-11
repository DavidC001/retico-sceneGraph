# retico-sceneGraph

A Retico extension for real-time scene graph generation, embedding, memory management, and visualization using the RelTR model.

## Features
- Generate scene graphs from images, extracting objects and their relationships.
- Embed triplets for Retrieval-Augmented Generation (RAG) queries.
- Store and query scene graphs in a memory module across multiple camera sources.
- Visualize scene graphs on images with bounding boxes and relationship arrows.

To tag different camera sources, use the `meta_data` field when creating the video producing module and populate `camera_name` with the camera identifier. This will allow the scene graph module to differentiate between inputs from different cameras.

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
