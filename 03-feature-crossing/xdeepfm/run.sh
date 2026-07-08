#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

echo "[xDeepFM] MovieLens 特征交叉实验"
echo "[xDeepFM] 使用 CIN 显式学习高阶特征交叉，同时保留 DNN 的非线性表达。"
echo "[xDeepFM] 默认使用全量数据；如果要快速试跑，可以传 --sample-ratings 2000000。"
echo "[xDeepFM] 默认 DataLoader workers: 8。"
echo "[xDeepFM] 默认保存 best.pt；如果完全不想写 .pt，可以传 --no-save-checkpoints。"

PYTHONPATH="../..:.." python main.py "$@"

