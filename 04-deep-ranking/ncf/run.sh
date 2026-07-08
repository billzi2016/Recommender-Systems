#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

echo "[NCF] MovieLens 深度精排实验"
echo "[NCF] 使用用户 ID 和电影 ID embedding，通过 MLP 预测用户是否喜欢。"
echo "[NCF] 默认使用全量数据；如果要快速试跑，可以传 --sample-ratings 2000000。"
echo "[NCF] 默认 DataLoader workers: 8。"
echo "[NCF] 默认保存 best.pt；如果完全不想写 .pt，可以传 --no-save-checkpoints。"

PYTHONPATH="../..:.." python main.py "$@"

