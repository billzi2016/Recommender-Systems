#!/usr/bin/env bash
set -euo pipefail

echo "[TwoTower] Start PyTorch two tower retrieval."
echo "[TwoTower] 步骤：读取 MovieLens -> 时间切分 -> 训练双塔 -> early stopping -> 向量召回 -> 写报告。"
echo "[TwoTower] 默认最多训练 1000 轮，但会根据验证集 loss early stopping。"
echo "[TwoTower] 默认 DataLoader workers: 8。"
echo "[TwoTower] 默认保存 best.pt；主 README 推荐 --checkpoint-every 0，只保存 best。"
echo "[TwoTower] 输出：report.md 和 report.zh.md"

cd "$(dirname "$0")"
PYTHONPATH="../../" python main.py "$@"
