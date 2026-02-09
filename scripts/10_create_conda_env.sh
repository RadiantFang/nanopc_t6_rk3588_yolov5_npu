#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${ENV_NAME:-rk3588_yolov5}"
PY_VER="${PY_VER:-3.8}"

if ! command -v conda >/dev/null 2>&1; then
  echo "[ERR] conda not found"
  echo "[TIP] install Miniconda/Anaconda first"
  exit 1
fi

eval "$(conda shell.bash hook)"

if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  echo "[INFO] conda env already exists: ${ENV_NAME}"
else
  echo "[INFO] creating conda env: ${ENV_NAME} (python=${PY_VER})"
  conda create -y -n "${ENV_NAME}" python="${PY_VER}"
fi

conda activate "${ENV_NAME}"

echo "[INFO] upgrading pip"
python -m pip install --upgrade pip setuptools wheel

echo "[INFO] installing base dependencies (pip)"
python -m pip install \
  numpy==1.24.4 \
  opencv-python \
  scipy \
  tqdm \
  psutil \
  ruamel.yaml \
  protobuf==3.20.3

echo "[INFO] installing optional dependencies"
if ! python -m pip install onnx==1.14.1; then
  echo "[WARN] onnx install failed, skip"
fi
if ! python -m pip install onnxruntime==1.16.0; then
  echo "[WARN] onnxruntime install failed, skip"
fi

if [[ -n "${RKNN_TOOLKIT2_WHL:-}" ]]; then
  echo "[INFO] installing rknn-toolkit2: ${RKNN_TOOLKIT2_WHL}"
  python -m pip install "${RKNN_TOOLKIT2_WHL}"
else
  echo "[WARN] RKNN_TOOLKIT2_WHL not set, skip rknn-toolkit2 install"
fi

if [[ -n "${RKNN_LITE2_WHL:-}" ]]; then
  echo "[INFO] installing rknn-toolkit-lite2: ${RKNN_LITE2_WHL}"
  python -m pip install "${RKNN_LITE2_WHL}"
else
  echo "[WARN] RKNN_LITE2_WHL not set, skip rknn-toolkit-lite2 install"
fi

echo "[DONE] conda env ready: ${ENV_NAME}"
echo "[NEXT] run: conda activate ${ENV_NAME}"
