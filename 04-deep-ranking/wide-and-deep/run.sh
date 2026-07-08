#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

echo "[Wide&Deep] MovieLens 深度精排实验"
echo "[Wide&Deep] Wide 侧记规则，Deep 侧用 embedding 学泛化。"
echo "[Wide&Deep] 默认使用全量数据；如果要快速试跑，可以传 --sample-ratings 2000000。"
echo "[Wide&Deep] 默认 DataLoader workers: 8。"
echo "[Wide&Deep] 默认保存 best.pt；如果完全不想写 .pt，可以传 --no-save-checkpoints。"

PYTHONPATH="../..:.." python main.py "$@"

