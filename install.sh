#!/bin/bash
# Install PyTorch and related packages first
pip install torch==2.9.0+cu126 torchvision==0.24.0+cu126 torchaudio==2.9.0+cu126 --extra-index-url https://download.pytorch.org/whl/cu126

# Install everything else
pip install -r requirements.txt