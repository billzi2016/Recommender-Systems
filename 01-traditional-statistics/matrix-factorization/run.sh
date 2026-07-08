#!/usr/bin/env bash
set -euo pipefail

echo "[MF] Start matrix factorization."
echo "[MF] PyTorch 会自动选择 cuda / mps / cpu。Mac 上可使用 MPS。"
echo "[MF] 步骤：读取 MovieLens -> 时间切分 -> 训练 embedding -> early stopping -> 写报告。"
echo "[MF] 默认最多训练 1000 轮，但会根据验证集 RMSE early stopping。"
echo "[MF] 默认保存 best.pt，并只保留少量中间 checkpoint；不想保存时加 --no-save-checkpoints。"
echo "[MF] 输出：report.md 和 report.zh.md"

cd "$(dirname "$0")"
PYTHONPATH="../../" python main.py "$@"
