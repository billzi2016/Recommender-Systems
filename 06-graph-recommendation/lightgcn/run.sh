#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

echo "[LightGCN] MovieLens 图推荐实验"
echo "[LightGCN] 使用 rating >= 4.0 的用户-电影边构建二部图，并用 BPR loss 训练。"
echo "[LightGCN] 默认使用全量数据；如果要快速试跑，可以传 --sample-ratings 2000000。"
echo "[LightGCN] 默认 DataLoader workers: 8。"
echo "[LightGCN] 默认保存 best.pt；如果完全不想写 .pt，可以传 --no-save-checkpoints。"

PYTHONPATH="../..:.." python main.py "$@"

