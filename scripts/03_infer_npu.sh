#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY_DIR="${ROOT_DIR}/rknn_model_zoo/examples/yolov5/python"
MODEL_DIR="${ROOT_DIR}/rknn_model_zoo/examples/yolov5/model"
OUT_DIR="${ROOT_DIR}/nanopc_t6_rk3588_yolov5_npu/output"

TARGET="${TARGET:-rk3588}"
RKNN_PATH="${RKNN_PATH:-${OUT_DIR}/yolov5_rk3588.rknn}"
IMG_FOLDER="${IMG_FOLDER:-${MODEL_DIR}}"
CORE_MASK="${CORE_MASK:-NPU_CORE_0_1_2}"
NPU_LOAD_PATH="${NPU_LOAD_PATH:-/sys/kernel/debug/rknpu/load}"
NPU_CUR_FREQ_PATH="${NPU_CUR_FREQ_PATH:-/sys/class/devfreq/fdab0000.npu/cur_freq}"
NPU_GOVERNOR_PATH="${NPU_GOVERNOR_PATH:-/sys/class/devfreq/fdab0000.npu/governor}"
NPU_MIN_FREQ_PATH="${NPU_MIN_FREQ_PATH:-/sys/class/devfreq/fdab0000.npu/min_freq}"
NPU_MAX_FREQ_PATH="${NPU_MAX_FREQ_PATH:-/sys/class/devfreq/fdab0000.npu/max_freq}"
NPU_SET_FREQ="${NPU_SET_FREQ:-1}"
NPU_DEFAULT_FREQ="${NPU_DEFAULT_FREQ:-800000000}"
NPU_MONITOR_INTERVAL="${NPU_MONITOR_INTERVAL:-0.2}"
NPU_MONITOR_LOG="${NPU_MONITOR_LOG:-${OUT_DIR}/npu_load.log}"

if [[ -z "${CONDA_DEFAULT_ENV:-}" ]]; then
  echo "[ERR] conda env is not active"
  echo "[TIP] run: conda activate rk3588_yolov5"
  exit 1
fi

if [[ ! -f "${RKNN_PATH}" ]]; then
  echo "[ERR] RKNN model not found: ${RKNN_PATH}"
  echo "[TIP] run convert first: bash nanopc_t6_rk3588_yolov5_npu/scripts/02_convert_rknn.sh"
  exit 1
fi

if ! find "${IMG_FOLDER}" -maxdepth 1 -type f \( \
  -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" -o -iname "*.bmp" -o -iname "*.webp" -o -iname "*.tif" -o -iname "*.tiff" \
\) | grep -q .; then
  echo "[ERR] no input images found in ${IMG_FOLDER}"
  echo "[TIP] run: bash nanopc_t6_rk3588_yolov5_npu/scripts/01_download_model.sh"
  exit 1
fi

cd "${PY_DIR}"

sys_write() {
  local path="$1"
  local value="$2"
  if [[ -w "${path}" ]]; then
    printf "%s" "${value}" > "${path}"
    return 0
  fi
  if command -v sudo >/dev/null 2>&1 && sudo -n true >/dev/null 2>&1; then
    printf "%s" "${value}" | sudo tee "${path}" >/dev/null
    return 0
  fi
  return 1
}

read_value() {
  local path="$1"
  if [[ -r "${path}" ]]; then
    cat "${path}" 2>/dev/null || true
    return 0
  fi
  if command -v sudo >/dev/null 2>&1 && sudo -n true >/dev/null 2>&1; then
    sudo cat "${path}" 2>/dev/null || true
    return 0
  fi
  return 1
}

if [[ "${NPU_SET_FREQ}" == "1" ]]; then
  if sys_write "${NPU_GOVERNOR_PATH}" "userspace" \
    && sys_write "${NPU_MIN_FREQ_PATH}" "${NPU_DEFAULT_FREQ}" \
    && sys_write "${NPU_MAX_FREQ_PATH}" "${NPU_DEFAULT_FREQ}"; then
    echo "[INFO] NPU freq locked to ${NPU_DEFAULT_FREQ} Hz"
  else
    echo "[WARN] failed to set NPU freq, check sudo permission"
  fi
fi

cur_freq_now="$(read_value "${NPU_CUR_FREQ_PATH}" || true)"
if [[ -n "${cur_freq_now}" ]]; then
  echo "[INFO] NPU cur_freq: ${cur_freq_now} Hz"
fi

echo "[INFO] run NPU inference on ${TARGET}"

monitor_pid=""
cleanup_monitor() {
  if [[ -n "${monitor_pid}" ]]; then
    kill "${monitor_pid}" >/dev/null 2>&1 || true
    wait "${monitor_pid}" 2>/dev/null || true
  fi
}
trap cleanup_monitor EXIT

if read_value "${NPU_LOAD_PATH}" >/dev/null 2>&1; then
  : > "${NPU_MONITOR_LOG}"
  raw_now="$(read_value "${NPU_LOAD_PATH}" || true)"
  if [[ -n "${raw_now}" ]]; then
    echo "[INFO] rknpu/load raw: ${raw_now}"
  fi
  (
    while true; do
      now="$(date +%s.%N)"
      load_raw="$(read_value "${NPU_LOAD_PATH}" || true)"
      freq_raw="$(read_value "${NPU_CUR_FREQ_PATH}" || true)"
      core0="$(echo "${load_raw}" | grep -oE 'Core0:[[:space:]]*[0-9]+%' | grep -oE '[0-9]+' | tail -n1 || true)"
      core1="$(echo "${load_raw}" | grep -oE 'Core1:[[:space:]]*[0-9]+%' | grep -oE '[0-9]+' | tail -n1 || true)"
      core2="$(echo "${load_raw}" | grep -oE 'Core2:[[:space:]]*[0-9]+%' | grep -oE '[0-9]+' | tail -n1 || true)"
      load_pct="$(
        printf "%s\n%s\n%s\n" "${core0}" "${core1}" "${core2}" \
          | awk '/^[0-9]+$/{s+=$1; c+=1} END{if(c>0) printf "%.0f", s/c}'
      )"
      freq_hz="$(echo "${freq_raw}" | grep -oE '^[0-9]+' || true)"
      if [[ -n "${load_pct}" && -n "${freq_hz}" ]]; then
        printf "%s %s %s %s %s %s\n" "${now}" "${load_pct}" "${freq_hz}" "${core0:-0}" "${core1:-0}" "${core2:-0}" >> "${NPU_MONITOR_LOG}"
      fi
      sleep "${NPU_MONITOR_INTERVAL}"
    done
  ) &
  monitor_pid="$!"
  echo "[INFO] NPU monitor on: ${NPU_LOAD_PATH}"
fi

python3 yolov5.py --model_path "${RKNN_PATH}" --target "${TARGET}" --core_mask "${CORE_MASK}" --img_folder "${IMG_FOLDER}" --img_save

cleanup_monitor
monitor_pid=""

if [[ -s "${NPU_MONITOR_LOG}" ]]; then
  awk '
    {
      if ($2 ~ /^[0-9]+$/) {
        c += 1
        sum += $2
        if ($2 > max) max = $2
      }
      if ($3 ~ /^[0-9]+$/) {
        fsum += $3
      }
      if ($4 ~ /^[0-9]+$/) { c0sum += $4; if ($4 > c0max) c0max = $4 }
      if ($5 ~ /^[0-9]+$/) { c1sum += $5; if ($5 > c1max) c1max = $5 }
      if ($6 ~ /^[0-9]+$/) { c2sum += $6; if ($6 > c2max) c2max = $6 }
    }
    END {
      if (c > 0) {
        printf "[INFO] NPU load samples: %d\n", c
        printf "[INFO] NPU load avg: %.2f%%\n", sum / c
        printf "[INFO] NPU load peak: %d%%\n", max
        if (fsum > 0) {
          printf "[INFO] NPU freq avg: %.0f Hz\n", fsum / c
        }
        printf "[INFO] Core0 avg/peak: %.2f%%/%d%%\n", c0sum / c, c0max
        printf "[INFO] Core1 avg/peak: %.2f%%/%d%%\n", c1sum / c, c1max
        printf "[INFO] Core2 avg/peak: %.2f%%/%d%%\n", c2sum / c, c2max
      }
    }
  ' "${NPU_MONITOR_LOG}"
  echo "[INFO] NPU load log: ${NPU_MONITOR_LOG}"
else
  echo "[WARN] NPU load monitor data not collected"
fi

echo "[DONE] result image: ${PY_DIR}/result.jpg"
