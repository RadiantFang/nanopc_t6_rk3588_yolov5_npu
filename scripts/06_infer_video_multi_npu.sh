#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PROJ_DIR="${ROOT_DIR}/nanopc_t6_rk3588_yolov5_npu"
MODEL_DIR="${ROOT_DIR}/rknn_model_zoo/examples/yolov5/model"

TARGET="${TARGET:-rk3588}"
RKNN_PATH="${RKNN_PATH:-${PROJ_DIR}/output/yolov5_rk3588.rknn}"
VIDEO_IN="${VIDEO_IN:-${PROJ_DIR}/output/test_video.mp4}"
VIDEO_OUT="${VIDEO_OUT:-${PROJ_DIR}/output/test_video_result_multi.mp4}"
ANCHORS="${ANCHORS:-${MODEL_DIR}/anchors_yolov5.txt}"
MAX_FRAMES="${MAX_FRAMES:-300}"
NPU_DEFAULT_FREQ="${NPU_DEFAULT_FREQ:-800000000}"
MONITOR_INTERVAL="${MONITOR_INTERVAL:-0.2}"
LOAD_LOG="${LOAD_LOG:-${PROJ_DIR}/output/multi_rknpu_load.log}"
CORE_MASKS="${CORE_MASKS:-NPU_CORE_0,NPU_CORE_1,NPU_CORE_2}"
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

mkdir -p "${PROJ_DIR}/output"

if command -v sudo >/dev/null 2>&1 && sudo -n true >/dev/null 2>&1; then
  echo userspace | sudo tee /sys/class/devfreq/fdab0000.npu/governor >/dev/null
  echo "${NPU_DEFAULT_FREQ}" | sudo tee /sys/class/devfreq/fdab0000.npu/min_freq >/dev/null
  echo "${NPU_DEFAULT_FREQ}" | sudo tee /sys/class/devfreq/fdab0000.npu/max_freq >/dev/null
  cur_freq="$(cat /sys/class/devfreq/fdab0000.npu/cur_freq 2>/dev/null || true)"
  echo "[INFO] NPU cur_freq: ${cur_freq} Hz"
else
  echo "[WARN] sudo non-interactive unavailable, skip NPU freq lock"
fi

: > "${LOAD_LOG}"

(
  while true; do
    line="$(sudo cat /sys/kernel/debug/rknpu/load 2>/dev/null || true)"
    if [[ -n "${line}" ]]; then
      echo "$(date +%s.%N) ${line}" >> "${LOAD_LOG}"
    fi
    sleep "${MONITOR_INTERVAL}"
  done
) & mon_pid=$!

python3 "${PROJ_DIR}/scripts/infer_video_rknn_multi.py" \
  --model_path "${RKNN_PATH}" \
  --target "${TARGET}" \
  --core_masks "${CORE_MASKS}" \
  --input "${VIDEO_IN}" \
  --output "${VIDEO_OUT}" \
  --anchors "${ANCHORS}" \
  --crop_mode "${CROP_MODE}" \
  --crop_ratio "${CROP_RATIO}" \
  $( [[ "${AUTO_TRANSCODE}" == "1" ]] && echo "--auto_transcode" ) \
  --max_frames "${MAX_FRAMES}" 2>&1 | python3 -u -c '
import re, sys

patterns = [
    re.compile(r"^I rknn-toolkit2 version:"),
    re.compile(r"^--> Init runtime environment$"),
    re.compile(r"^I target set by user is:"),
    re.compile(r"^done$"),
    re.compile(r"^Model-.* is rknn model, starting val$"),
]
seen = set()
buf = ""
while True:
    ch = sys.stdin.read(1)
    if ch == "":
        if buf:
            s = buf
            is_dup_line = any(p.search(s) for p in patterns)
            if (not is_dup_line) or (s not in seen):
                if is_dup_line:
                    seen.add(s)
                sys.stdout.write(s)
                sys.stdout.flush()
        break
    if ch == "\r":
        sys.stdout.write(buf + "\r")
        sys.stdout.flush()
        buf = ""
        continue
    if ch == "\n":
        s = buf
        buf = ""
        is_dup_line = any(p.search(s) for p in patterns)
        if is_dup_line and s in seen:
            continue
        if is_dup_line:
            seen.add(s)
        sys.stdout.write(s + "\n")
        sys.stdout.flush()
        continue
    buf += ch
'

kill "${mon_pid}" >/dev/null 2>&1 || true
wait "${mon_pid}" 2>/dev/null || true

awk '
  {
    c0=0; c1=0; c2=0
    if (match($0, /Core0:[[:space:]]*[0-9]+%/)) {
      t=substr($0, RSTART, RLENGTH); split(t, a, ":"); gsub(/[^0-9]/, "", a[2]); c0=a[2]+0
    }
    if (match($0, /Core1:[[:space:]]*[0-9]+%/)) {
      t=substr($0, RSTART, RLENGTH); split(t, a, ":"); gsub(/[^0-9]/, "", a[2]); c1=a[2]+0
    }
    if (match($0, /Core2:[[:space:]]*[0-9]+%/)) {
      t=substr($0, RSTART, RLENGTH); split(t, a, ":"); gsub(/[^0-9]/, "", a[2]); c2=a[2]+0
    }
    s0+=c0; s1+=c1; s2+=c2
    if (c0>p0) p0=c0; if (c1>p1) p1=c1; if (c2>p2) p2=c2
    n+=1
  }
  END {
    if (n > 0) {
      printf("[INFO] samples=%d\n", n)
      printf("[INFO] Core0 avg/peak: %.2f%%/%d%%\n", s0/n, p0)
      printf("[INFO] Core1 avg/peak: %.2f%%/%d%%\n", s1/n, p1)
      printf("[INFO] Core2 avg/peak: %.2f%%/%d%%\n", s2/n, p2)
    } else {
      print("[WARN] no load samples")
    }
  }
' "${LOAD_LOG}"

echo "[INFO] load log: ${LOAD_LOG}"
ls -lh "${VIDEO_OUT}"
