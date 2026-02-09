#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY_DIR="${ROOT_DIR}/rknn_model_zoo/examples/yolov5/python"
MODEL_DIR="${ROOT_DIR}/rknn_model_zoo/examples/yolov5/model"
OUT_DIR="${ROOT_DIR}/nanopc_t6_rk3588_yolov5_npu/output"

mkdir -p "${OUT_DIR}"

ONNX_PATH="${ONNX_PATH:-${MODEL_DIR}/yolov5m.onnx}"
TARGET="${TARGET:-rk3588}"
DTYPE="${DTYPE:-i8}"
RKNN_OUT="${RKNN_OUT:-${OUT_DIR}/yolov5_rk3588.rknn}"

if [[ -z "${CONDA_DEFAULT_ENV:-}" ]]; then
  echo "[ERR] conda env is not active"
  echo "[TIP] run: conda activate rk3588_yolov5"
  exit 1
fi

if [[ ! -f "${ONNX_PATH}" ]]; then
  echo "[ERR] ONNX not found: ${ONNX_PATH}"
  echo "[TIP] run: bash nanopc_t6_rk3588_yolov5_npu/scripts/01_download_model.sh"
  exit 1
fi

cd "${PY_DIR}"

echo "[INFO] convert ONNX -> RKNN"
echo "[INFO] ONNX_PATH=${ONNX_PATH}"
echo "[INFO] TARGET=${TARGET}, DTYPE=${DTYPE}"
echo "[INFO] RKNN_OUT=${RKNN_OUT}"

python3 convert.py "${ONNX_PATH}" "${TARGET}" "${DTYPE}" "${RKNN_OUT}"

echo "[DONE] generated: ${RKNN_OUT}"
