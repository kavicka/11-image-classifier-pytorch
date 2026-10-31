#!/bin/sh
# Every long stage in order, about 5 hours on an M4 Pro. Detached, so it outlives the terminal:
#   nohup caffeinate -i sh run_all.sh > train.log 2>&1 &
set -e
uv run python -u experiment_cifar.py
uv run python -u experiment_food.py
uv run python -u gradcam.py
uv run python -u export_onnx.py
uv run python plot.py
echo ALL_DONE
