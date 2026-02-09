#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PROJ_DIR="${ROOT_DIR}/nanopc_t6_rk3588_yolov5_npu"

printf "[INFO] Root: %s\n" "${ROOT_DIR}"
printf "[INFO] Project: %s\n" "${PROJ_DIR}"

if [[ -d "${ROOT_DIR}/rknn_model_zoo" ]]; then
  echo "[OK] rknn_model_zoo repo exists"
else
  echo "[ERR] missing ${ROOT_DIR}/rknn_model_zoo"
  exit 1
fi

echo "[INFO] uname -a"
uname -a

echo "[INFO] Python"
python3 --version || true

if [[ -n "${CONDA_DEFAULT_ENV:-}" ]]; then
  echo "[OK] conda env active: ${CONDA_DEFAULT_ENV}"
else
  echo "[WARN] conda env is not active"
  echo "[TIP] run: conda activate rk3588_yolov5"
fi

echo "[INFO] check rknn python packages"
python3 - <<'PY'
import importlib
mods = ["rknn", "rknnlite"]
for m in mods:
    try:
        importlib.import_module(m)
        print(f"[OK] {m} import success")
    except Exception as e:
        print(f"[WARN] {m} import failed: {e}")
PY

echo "[DONE] environment check finished"
