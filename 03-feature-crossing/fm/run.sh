#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

echo "[FM] MovieLens 特征交叉实验"
echo "[FM] 使用用户 ID、电影 ID、genres、时间段训练因子分解机。"
echo "[FM] 默认使用全量数据；如果要快速试跑，可以传 --sample-ratings 2000000。"
echo "[FM] 默认 DataLoader workers: 8。"
echo "[FM] 默认保存 best.pt；如果完全不想写 .pt，可以传 --no-save-checkpoints。"

PYTHONPATH="../..:.." python main.py "$@"

