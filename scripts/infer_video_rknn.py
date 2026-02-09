#!/usr/bin/env python3
import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np


def print_progress(done, total, prefix="[INFO] processed", width=36):
    if total > 0:
        ratio = min(max(done / total, 0.0), 1.0)
        filled = int(width * ratio)
        bar = "#" * filled + "-" * (width - filled)
        msg = f"\r{prefix} [{bar}] {done}/{total} ({ratio * 100:.1f}%)"
    else:
        msg = f"\r{prefix} {done} frames"
    sys.stdout.write(msg)
    sys.stdout.flush()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True, type=str)
    p.add_argument("--target", default="rk3588", type=str)
    p.add_argument("--device_id", default=None, type=str)
    p.add_argument("--core_mask", default="NPU_CORE_0_1_2", type=str)
    p.add_argument("--input", required=True, type=str)
    p.add_argument("--output", required=True, type=str)
    p.add_argument("--anchors", required=True, type=str)
    p.add_argument("--max_frames", default=0, type=int, help="0 means all frames")
    p.add_argument("--crop_mode", default="none", choices=["none", "center"], type=str)
    p.add_argument("--crop_ratio", default=1.0, type=float)
    p.add_argument("--auto_transcode", action="store_true", default=True)
    return p.parse_args()


def draw_boxes(image, boxes, scores, classes, class_names):
    for box, score, cls in zip(boxes, scores, classes):
        x1, y1, x2, y2 = [int(v) for v in box]
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(
            image,
            f"{class_names[int(cls)].strip()} {score:.2f}",
            (x1, max(15, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
        )

def preprocess_frame(frame, crop_mode="none", crop_ratio=1.0):
    if frame is None:
        return None
    if frame.ndim == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    elif frame.ndim == 3 and frame.shape[2] == 4:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    elif frame.ndim != 3 or frame.shape[2] != 3:
        raise RuntimeError(f"preprocess failed: unsupported frame shape {frame.shape}")
    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)

    if crop_mode == "center" and 0 < crop_ratio < 1.0:
        h, w = frame.shape[:2]
        ch = max(1, int(h * crop_ratio))
        cw = max(1, int(w * crop_ratio))
        y0 = max(0, (h - ch) // 2)
        x0 = max(0, (w - cw) // 2)
        frame = frame[y0:y0 + ch, x0:x0 + cw]
    return frame


def open_video_with_fallback(input_path, auto_transcode=True):
    cap = cv2.VideoCapture(input_path)
    if cap.isOpened():
        return cap, input_path
    cap.release()

    if not auto_transcode:
        raise RuntimeError(f"cannot open input video: {input_path}")

    ffmpeg_bin = shutil.which("ffmpeg")
    if ffmpeg_bin is None:
        raise RuntimeError(f"cannot open input video: {input_path}, and ffmpeg not found for auto transcode")

    src = Path(input_path)
    converted = src.with_name(src.stem + "_transcoded.mp4")
    cmd = [
        ffmpeg_bin, "-y", "-i", str(src),
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "128k",
        str(converted),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    cap = cv2.VideoCapture(str(converted))
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"auto transcode failed for input video: {input_path}")
    print(f"[INFO] auto transcoded video: {converted}")
    return cap, str(converted)


def main():
    args = parse_args()
    if not (0 < args.crop_ratio <= 1.0):
        raise RuntimeError(f"invalid crop_ratio {args.crop_ratio}, expected (0,1]")

    proj_dir = Path(__file__).resolve().parents[1]
    root_dir = proj_dir.parent
    yolo_py_dir = root_dir / "rknn_model_zoo" / "examples" / "yolov5" / "python"

    if not yolo_py_dir.exists():
        raise FileNotFoundError(f"missing {yolo_py_dir}")

    sys.path.insert(0, str(yolo_py_dir))

    import yolov5 as y5
    from py_utils.coco_utils import COCO_test_helper

    with open(args.anchors, "r", encoding="utf-8") as f:
        values = [float(v) for v in f.readlines()]
        anchors = np.array(values).reshape(3, -1, 2).tolist()

    model_args = argparse.Namespace(
        model_path=args.model_path,
        target=args.target,
        device_id=args.device_id,
        core_mask=args.core_mask,
    )
    model, platform = y5.setup_model(model_args)
    if platform != "rknn":
        raise RuntimeError("video script expects .rknn model")

    cap, input_used = open_video_with_fallback(args.input, args.auto_transcode)
    if input_used != args.input:
        print(f"[INFO] use transcoded input: {input_used}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames_src = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if not fps or fps <= 1e-3:
        fps = 25.0
    if args.max_frames > 0:
        total_frames_plan = min(args.max_frames, total_frames_src) if total_frames_src > 0 else args.max_frames
    else:
        total_frames_plan = total_frames_src
    if total_frames_plan > 0:
        print(f"[INFO] total frames: {total_frames_plan}")
    else:
        print("[INFO] total frames: unknown")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.output, fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"cannot open output video: {args.output}")

    helper = COCO_test_helper(enable_letter_box=True)

    frame_idx = 0
    infer_time = 0.0
    t_all = time.time()

    # Startup check: run one-frame warmup and validate model input/output shape/layout.
    ret, first_frame = cap.read()
    if not ret:
        raise RuntimeError("startup check failed: cannot read first frame from input video")

    helper.letter_box_info_list = []
    first_frame = preprocess_frame(first_frame, args.crop_mode, args.crop_ratio)
    first_img = helper.letter_box(im=first_frame.copy(), new_shape=(y5.IMG_SIZE[1], y5.IMG_SIZE[0]), pad_color=(0, 0, 0))
    first_img = cv2.cvtColor(first_img, cv2.COLOR_BGR2RGB)
    t0 = time.time()
    first_outputs = model.run([first_img])
    infer_time += time.time() - t0
    y5.validate_model_io(first_img, first_outputs)
    print("[INFO] startup check passed: model input/output shape/layout is valid")

    # Rewind so formal inference always starts from frame 0.
    if not cap.set(cv2.CAP_PROP_POS_FRAMES, 0):
        cap.release()
        cap = cv2.VideoCapture(args.input)
        if not cap.isOpened():
            raise RuntimeError(f"cannot reopen input video after startup check: {args.input}")

    while True:
        if args.max_frames > 0 and frame_idx >= args.max_frames:
            break
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1

        frame = preprocess_frame(frame, args.crop_mode, args.crop_ratio)
        helper.letter_box_info_list = []
        img = helper.letter_box(im=frame.copy(), new_shape=(y5.IMG_SIZE[1], y5.IMG_SIZE[0]), pad_color=(0, 0, 0))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        t0 = time.time()
        outputs = model.run([img])
        infer_time += time.time() - t0

        boxes, classes, scores = y5.post_process(outputs, anchors)

        vis = frame.copy()
        if boxes is not None:
            real_boxes = helper.get_real_box(boxes)
            draw_boxes(vis, real_boxes, scores, classes, y5.CLASSES)

        cv2.putText(
            vis,
            f"frame:{frame_idx}",
            (10, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
        )

        if vis.shape[1] != width or vis.shape[0] != height:
            vis = cv2.resize(vis, (width, height), interpolation=cv2.INTER_LINEAR)
        writer.write(vis)

        if frame_idx % 5 == 0 or (total_frames_plan > 0 and frame_idx == total_frames_plan):
            print_progress(frame_idx, total_frames_plan)

    print_progress(frame_idx, total_frames_plan)
    print()

    cap.release()
    writer.release()
    model.release()

    t_cost = time.time() - t_all
    avg_ms = (infer_time / max(frame_idx, 1)) * 1000.0
    fps_total = frame_idx / max(t_cost, 1e-6)

    print(f"[DONE] frames={frame_idx}")
    print(f"[DONE] avg_infer={avg_ms:.2f} ms/frame")
    print(f"[DONE] throughput={fps_total:.2f} fps")
    print(f"[DONE] output={args.output}")


if __name__ == "__main__":
    main()
