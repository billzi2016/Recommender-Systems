#!/usr/bin/env bash
set -euo pipefail

echo "[User-CF] Start user-based collaborative filtering."
echo "[User-CF] 步骤：读取 MovieLens -> 训练用户近邻 -> 生成推荐 -> 写报告。"
echo "[User-CF] 输出：report.md 和 report.zh.md"

cd "$(dirname "$0")"
PYTHONPATH="../../" python main.py "$@"
