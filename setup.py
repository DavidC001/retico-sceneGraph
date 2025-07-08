#!/usr/bin/env python

"""
Setup script for retico-objectFeatures
"""

import os
from setuptools import setup, find_packages

# Read version from version.py
version = {}
with open(os.path.join("retico_sceneGraph", "version.py")) as fp:
    exec(fp.read(), version)

# Read README for long description
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="retico-sceneGraph",
    version=version["__version__"],
    author="DavidC001",
    author_email="",
    description="A ReTiCo module for scene graph generation.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    python_requires=">=3.7",
    install_requires=[
        # "retico-core",
        # "retico-vision",
        "sentence_transformers>=2.7.0",
        "transformers>=4.51.0",
        "torch",
        "torchvision",
        "bitsandbytes",
        "networkx",
    ],
    keywords="retico, real-time, dialogue, object detection, computer vision, CLIP, feature extraction",
    project_urls={
        
    },
)