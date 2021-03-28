#!/bin/sh
git submodule init
git submodule update
ln tablut.py muzero-general/games/tablut.py
conda create -n muzero python=3.7
conda activate muzero
python -m pip install -r muzero-general/requirements.txt