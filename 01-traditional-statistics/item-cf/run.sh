#!/usr/bin/env bash
set -euo pipefail

echo "[Item-CF] Start item-based collaborative filtering."
echo "[Item-CF] It will load MovieLens, build item neighbors, print tqdm progress, and write reports."
echo "[Item-CF] Output files: report.md and report.zh.md"
echo "[Item-CF] If raw/ml-32m is missing, the shared loader will unzip raw/ml-32m.zip automatically."

cd "$(dirname "$0")"
PYTHONPATH="../../" python main.py "$@"
