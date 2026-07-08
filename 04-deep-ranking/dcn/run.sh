#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

echo "[DCN] MovieLens 深度精排实验"
echo "[DCN] 使用 cross layer 显式学习用户、电影、genre、时间段之间的交叉。"
echo "[DCN] 默认使用全量数据；如果要快速试跑，可以传 --sample-ratings 2000000。"
echo "[DCN] 默认 DataLoader workers: 8。"
echo "[DCN] 默认保存 best.pt；如果完全不想写 .pt，可以传 --no-save-checkpoints。"

PYTHONPATH="../..:.." python main.py "$@"

