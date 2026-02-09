#!/usr/bin/env python3
import argparse
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True, type=str)
    p.add_argument("--target", default="rk3588", type=str)
    p.add_argument("--device_id", default=None, type=str)
    p.add_argument("--core_masks", default="NPU_CORE_0,NPU_CORE_1,NPU_CORE_2", type=str)
    p.add_argument("--input", required=True, type=str)
    p.add_argument("--output", required=True, type=str)
    p.add_argument("--anchors", required=True, type=str)
    p.add_argument("--max_frames", default=0, type=int)
    p.add_argument("--queue_size", default=16, type=int)
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


def print_progress(done, total, prefix="[INFO] merged", width=36):
    if total > 0:
        ratio = min(max(done / total, 0.0), 1.0)
        filled = int(width * ratio)
        bar = "#" * filled + "-" * (width - filled)
        msg = f"\r{prefix} [{bar}] {done}/{total} ({ratio * 100:.1f}%)"
    else:
        msg = f"\r{prefix} {done} frames"
    sys.stdout.write(msg)
    sys.stdout.flush()

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
    core_masks = [c.strip() for c in args.core_masks.split(",") if c.strip()]
    if not core_masks:
        raise ValueError("core_masks is empty")

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

    # Startup check: run one-frame warmup and validate model input/output shape/layout.
    ret, check_frame = cap.read()
    if not ret:
        raise RuntimeError("startup check failed: cannot read first frame from input video")
    check_helper = COCO_test_helper(enable_letter_box=True)
    check_frame = preprocess_frame(check_frame, args.crop_mode, args.crop_ratio)
    check_helper.letter_box_info_list = []
    check_img = check_helper.letter_box(
        im=check_frame.copy(),
        new_shape=(y5.IMG_SIZE[1], y5.IMG_SIZE[0]),
        pad_color=(0, 0, 0),
    )
    check_img = cv2.cvtColor(check_img, cv2.COLOR_BGR2RGB)
    check_args = argparse.Namespace(
        model_path=args.model_path,
        target=args.target,
        device_id=args.device_id,
        core_mask=core_masks[0],
    )
    check_model, check_platform = y5.setup_model(check_args)
    if check_platform != "rknn":
        raise RuntimeError("startup check failed: model is not rknn")
    check_outputs = check_model.run([check_img])
    y5.validate_model_io(check_img, check_outputs)
    check_model.release()
    print("[INFO] startup check passed: model input/output shape/layout is valid")

    # Seek back to beginning for normal processing.
    if not cap.set(cv2.CAP_PROP_POS_FRAMES, 0):
        cap.release()
        cap = cv2.VideoCapture(args.input)
        if not cap.isOpened():
            raise RuntimeError(f"cannot reopen input video after startup check: {args.input}")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"cannot open output video: {args.output}")

    task_queues = [queue.Queue(maxsize=args.queue_size) for _ in core_masks]
    result_queue = queue.Queue()

    stats = {
        i: {"frames": 0, "infer_time": 0.0, "core_mask": core_masks[i]} for i in range(len(core_masks))
    }

    def worker(worker_id):
        model_args = argparse.Namespace(
            model_path=args.model_path,
            target=args.target,
            device_id=args.device_id,
            core_mask=core_masks[worker_id],
        )
        model, platform = y5.setup_model(model_args)
        if platform != "rknn":
            raise RuntimeError("multi video script expects .rknn model")

        helper = COCO_test_helper(enable_letter_box=True)

        while True:
            item = task_queues[worker_id].get()
            if item is None:
                break
            frame_idx, frame = item

            frame = preprocess_frame(frame, args.crop_mode, args.crop_ratio)
            helper.letter_box_info_list = []
            img = helper.letter_box(
                im=frame.copy(),
                new_shape=(y5.IMG_SIZE[1], y5.IMG_SIZE[0]),
                pad_color=(0, 0, 0),
            )
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            t0 = time.time()
            outputs = model.run([img])
            infer_cost = time.time() - t0

            boxes, classes, scores = y5.post_process(outputs, anchors)
            vis = frame.copy()
            if boxes is not None:
                real_boxes = helper.get_real_box(boxes)
                draw_boxes(vis, real_boxes, scores, classes, y5.CLASSES)

            cv2.putText(vis, f"frame:{frame_idx}", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(
                vis,
                f"core:{core_masks[worker_id]}",
                (10, 52),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
            )

            if vis.shape[1] != width or vis.shape[0] != height:
                vis = cv2.resize(vis, (width, height), interpolation=cv2.INTER_LINEAR)
            stats[worker_id]["frames"] += 1
            stats[worker_id]["infer_time"] += infer_cost
            result_queue.put((frame_idx, vis))

        model.release()

    threads = [threading.Thread(target=worker, args=(i,), daemon=True) for i in range(len(core_masks))]
    for t in threads:
        t.start()

    t_all = time.time()
    frame_idx = 0
    submitted = 0
    next_to_write = 1
    buffer = {}
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if args.max_frames > 0 and submitted >= args.max_frames:
            break
        frame_idx += 1
        worker_id = (frame_idx - 1) % len(core_masks)
        task_queues[worker_id].put((frame_idx, frame))
        submitted += 1

        while True:
            try:
                idx, vis = result_queue.get_nowait()
            except queue.Empty:
                break
            buffer[idx] = vis
            while next_to_write in buffer:
                writer.write(buffer.pop(next_to_write))
                if next_to_write % 30 == 0:
                    print_progress(next_to_write, total_frames_plan if total_frames_plan > 0 else submitted)
                next_to_write += 1

    cap.release()

    for q in task_queues:
        q.put(None)

    pending = submitted - (next_to_write - 1)

    while pending > 0:
        idx, vis = result_queue.get()
        buffer[idx] = vis
        while next_to_write in buffer:
            writer.write(buffer.pop(next_to_write))
            if next_to_write % 30 == 0:
                print_progress(next_to_write, total_frames_plan if total_frames_plan > 0 else submitted)
            next_to_write += 1
            pending -= 1

    print_progress(submitted, total_frames_plan if total_frames_plan > 0 else submitted)
    print()

    for t in threads:
        t.join()

    writer.release()

    total_cost = time.time() - t_all
    print(f"[DONE] frames={submitted}")
    print(f"[DONE] merged_output={args.output}")
    print(f"[DONE] total_throughput={submitted / max(total_cost, 1e-6):.2f} fps")

    for i in range(len(core_masks)):
        f = stats[i]["frames"]
        it = stats[i]["infer_time"]
        avg_ms = (it / f * 1000.0) if f > 0 else 0.0
        print(f"[DONE] worker{i} core={stats[i]['core_mask']} frames={f} avg_infer={avg_ms:.2f} ms")


if __name__ == "__main__":
    main()
