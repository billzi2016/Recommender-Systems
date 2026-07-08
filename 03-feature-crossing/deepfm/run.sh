#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

echo "[DeepFM] MovieLens 特征交叉实验"
echo "[DeepFM] FM 学二阶交叉，MLP 学更复杂的非线性组合。"
echo "[DeepFM] 默认使用全量数据；如果要快速试跑，可以传 --sample-ratings 2000000。"
echo "[DeepFM] 默认 DataLoader workers: 8。"
echo "[DeepFM] 默认保存 best.pt；如果完全不想写 .pt，可以传 --no-save-checkpoints。"

PYTHONPATH="../..:.." python main.py "$@"

