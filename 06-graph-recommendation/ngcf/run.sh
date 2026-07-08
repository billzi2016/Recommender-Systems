#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

echo "[NGCF] MovieLens 图推荐实验"
echo "[NGCF] 使用用户-电影二部图、神经消息传递和 BPR loss 训练。"
echo "[NGCF] 默认使用全量数据；如果要快速试跑，可以传 --sample-ratings 2000000。"
echo "[NGCF] 默认 DataLoader workers: 8。"
echo "[NGCF] 默认保存 best.pt；如果完全不想写 .pt，可以传 --no-save-checkpoints。"

PYTHONPATH="../..:.." python main.py "$@"

