#!/bin/bash
conda create -n llava python=3.10 -y
conda activate llava
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
pip install --upgrade pip  -y && pip install -e . -y && conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
pip install -e ".[train]"
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
pip install flash-attn --no-build-isolation
