#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MODEL_DIR="${ROOT_DIR}/rknn_model_zoo/examples/yolov5/model"

cd "${MODEL_DIR}"

if [[ -f "yolov5s_relu.onnx" ]]; then
  echo "[INFO] yolov5s_relu.onnx already exists, skip download"
else
  echo "[INFO] downloading yolov5s_relu.onnx"
  bash download_model.sh
fi

if [[ -f "yolov5m.onnx" ]]; then
  echo "[INFO] yolov5m.onnx already exists, skip download"
else
  echo "[INFO] downloading yolov5m.onnx"
  wget -O ./yolov5m.onnx https://ftrg.zbox.filez.com/v2/delivery/data/95f00b0fc900458ba134f8b180b3f7a1/examples/yolov5/yolov5m.onnx
fi

if [[ ! -f "bus.jpg" ]]; then
  echo "[INFO] creating fallback test image: bus.jpg"
  python3 - <<'PY'
import numpy as np
import cv2

img = np.zeros((640, 640, 3), dtype=np.uint8)
img[:] = (32, 32, 32)
cv2.rectangle(img, (100, 120), (540, 520), (0, 255, 255), 3)
cv2.putText(img, "RK3588 YOLOv5 TEST", (90, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
cv2.imwrite("bus.jpg", img)
PY
  echo "[INFO] generated test image: bus.jpg"
fi

echo "[DONE] model assets ready in ${MODEL_DIR}"
