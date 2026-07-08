#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

echo "[GRU4Rec] MovieLens 序列推荐实验"
echo "[GRU4Rec] 使用高评分电影序列预测下一部电影。"
echo "[GRU4Rec] 默认使用全量数据；如果要快速试跑，可以传 --sample-ratings 2000000。"
echo "[GRU4Rec] 默认 DataLoader workers: 8。"
echo "[GRU4Rec] 默认保存 best.pt；如果完全不想写 .pt，可以传 --no-save-checkpoints。"

PYTHONPATH="../..:.." python main.py "$@"

