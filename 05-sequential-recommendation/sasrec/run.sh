#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

echo "[SASRec] MovieLens 序列推荐实验"
echo "[SASRec] 使用 causal self-attention 根据历史电影预测下一部电影。"
echo "[SASRec] 文档主线命令使用 --sample-ratings 2000000；全量数据请显式传 --sample-ratings none。"
echo "[SASRec] 默认 DataLoader workers: 8。"
echo "[SASRec] 默认保存 best.pt；如果完全不想写 .pt，可以传 --no-save-checkpoints。"

PYTHONPATH="../..:.." python main.py "$@"
