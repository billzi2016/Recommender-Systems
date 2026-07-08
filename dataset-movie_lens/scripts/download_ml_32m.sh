#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATASET_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
RAW_DIR="${DATASET_DIR}/raw"

mkdir -p "${RAW_DIR}"

curl -L \
  --fail \
  --show-error \
  --progress-bar \
  -o "${RAW_DIR}/ml-32m.zip" \
  "https://files.grouplens.org/datasets/movielens/ml-32m.zip"

curl -L \
  --fail \
  --show-error \
  --progress-bar \
  -o "${RAW_DIR}/ml-32m.zip.md5" \
  "https://files.grouplens.org/datasets/movielens/ml-32m.zip.md5"

if command -v md5sum >/dev/null 2>&1; then
  expected="$(awk '{print $1}' "${RAW_DIR}/ml-32m.zip.md5")"
  actual="$(md5sum "${RAW_DIR}/ml-32m.zip" | awk '{print $1}')"
elif command -v md5 >/dev/null 2>&1; then
  expected="$(awk '{print $1}' "${RAW_DIR}/ml-32m.zip.md5")"
  actual="$(md5 -q "${RAW_DIR}/ml-32m.zip")"
else
  echo "MD5 tool not found; skipped checksum verification."
  exit 0
fi

if [[ "${expected}" != "${actual}" ]]; then
  echo "Checksum failed: expected ${expected}, got ${actual}" >&2
  exit 1
fi

echo "Downloaded and verified ${RAW_DIR}/ml-32m.zip"
