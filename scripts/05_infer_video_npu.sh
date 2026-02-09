#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PROJ_DIR="${ROOT_DIR}/nanopc_t6_rk3588_yolov5_npu"
MODEL_DIR="${ROOT_DIR}/rknn_model_zoo/examples/yolov5/model"

TARGET="${TARGET:-rk3588}"
RKNN_PATH="${RKNN_PATH:-${PROJ_DIR}/output/yolov5_rk3588.rknn}"
VIDEO_IN="${VIDEO_IN:-${PROJ_DIR}/output/test_video.mp4}"
VIDEO_OUT="${VIDEO_OUT:-${PROJ_DIR}/output/test_video_result.mp4}"
ANCHORS="${ANCHORS:-${MODEL_DIR}/anchors_yolov5.txt}"
MAX_FRAMES="${MAX_FRAMES:-0}"
NPU_DEFAULT_FREQ="${NPU_DEFAULT_FREQ:-800000000}"
CORE_MASK="${CORE_MASK:-NPU_CORE_0_1_2}"
CROP_MODE="${CROP_MODE:-none}"
CROP_RATIO="${CROP_RATIO:-1.0}"
AUTO_TRANSCODE="${AUTO_TRANSCODE:-1}"

if [[ -z "${CONDA_DEFAULT_ENV:-}" ]]; then
  echo "[ERR] conda env is not active"
  echo "[TIP] run: conda activate rk3588_yolov5"
  exit 1
fi

if [[ ! -f "${RKNN_PATH}" ]]; then
  echo "[ERR] RKNN model not found: ${RKNN_PATH}"
  exit 1
fi

if [[ ! -f "${VIDEO_IN}" ]]; then
  echo "[ERR] input video not found: ${VIDEO_IN}"
  echo "[TIP] run: bash scripts/04_download_test_video.sh"
  exit 1
fi

if command -v sudo >/dev/null 2>&1 && sudo -n true >/dev/null 2>&1; then
  echo userspace | sudo tee /sys/class/devfreq/fdab0000.npu/governor >/dev/null
  echo "${NPU_DEFAULT_FREQ}" | sudo tee /sys/class/devfreq/fdab0000.npu/min_freq >/dev/null
  echo "${NPU_DEFAULT_FREQ}" | sudo tee /sys/class/devfreq/fdab0000.npu/max_freq >/dev/null
  cur_freq="$(cat /sys/class/devfreq/fdab0000.npu/cur_freq 2>/dev/null || true)"
  echo "[INFO] NPU cur_freq: ${cur_freq} Hz"
else
  echo "[WARN] sudo non-interactive unavailable, skip NPU freq lock"
fi

python3 "${PROJ_DIR}/scripts/infer_video_rknn.py" \
  --model_path "${RKNN_PATH}" \
  --target "${TARGET}" \
  --core_mask "${CORE_MASK}" \
  --input "${VIDEO_IN}" \
  --output "${VIDEO_OUT}" \
  --anchors "${ANCHORS}" \
  --crop_mode "${CROP_MODE}" \
  --crop_ratio "${CROP_RATIO}" \
  $( [[ "${AUTO_TRANSCODE}" == "1" ]] && echo "--auto_transcode" ) \
  --max_frames "${MAX_FRAMES}"

ls -lh "${VIDEO_OUT}"
