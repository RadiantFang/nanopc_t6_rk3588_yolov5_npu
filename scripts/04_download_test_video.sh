#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT_DIR="${ROOT_DIR}/nanopc_t6_rk3588_yolov5_npu/output"
VIDEO_PATH="${VIDEO_PATH:-${OUT_DIR}/test_video.mp4}"

mkdir -p "${OUT_DIR}"

if [[ -f "${VIDEO_PATH}" ]]; then
  echo "[INFO] video already exists: ${VIDEO_PATH}"
  exit 0
fi

URL1="https://raw.githubusercontent.com/mediaelement/mediaelement-files/master/big_buck_bunny.mp4"
URL2="https://samplelib.com/lib/preview/mp4/sample-5s.mp4"

echo "[INFO] downloading test video (mirror 1)"
if ! wget -O "${VIDEO_PATH}" "${URL1}"; then
  echo "[WARN] mirror 1 failed, trying mirror 2"
  wget -O "${VIDEO_PATH}" "${URL2}"
fi

echo "[DONE] video: ${VIDEO_PATH}"
ls -lh "${VIDEO_PATH}"
