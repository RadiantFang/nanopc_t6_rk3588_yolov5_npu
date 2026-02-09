# NanoPC T6 (RK3588) YOLOv5 NPU 项目说明

本项目用于在 **NanoPC T6 (RK3588)** 上部署 YOLOv5，并使用 Rockchip NPU（RKNN）进行推理。

> 注：本项目脚本与文档由 AI 协助完成。

## 0. 仓库拉取顺序（先依赖，后本项目）

```bash
cd <workspace-root>

# 1) 先拉取依赖仓库
git clone https://github.com/airockchip/rknn_model_zoo.git

# 2) 再拉取本项目
git clone git@github.com:RadiantFang/nanopc_t6_rk3588_yolov5_npu.git
```

目录关系：
- 本项目：`<workspace-root>/nanopc_t6_rk3588_yolov5_npu`
- 同级依赖：`<workspace-root>/rknn_model_zoo`
- 测试文件夹：`<workspace-root>/nanopc_t6_rk3588_yolov5_npu/test_assets`
  - `test_assets/bus.jpg`
  - `test_assets/che.mp4`

---

## 1. 功能概览

- 图片推理（RKNN）
- 视频单路推理（单输入单输出）
- 视频三核并发推理（单输入单输出）
- 启动前自动检查模型输入/输出 shape 与 layout
- 自动预处理（通道转换、类型转换、可选中心裁剪）
- 视频打不开时自动转码兜底（依赖 `ffmpeg`）
- NPU 频率控制（默认 800MHz）
- NPU 负载统计（若系统节点可读）

---

## 2. 脚本清单

- `scripts/00_check_env.sh`：环境检查
- `scripts/01_download_model.sh`：下载 ONNX，并在缺少测试图时自动生成 `bus.jpg`
- `scripts/02_convert_rknn.sh`：ONNX -> RKNN
- `scripts/03_infer_npu.sh`：图片推理（含 NPU 监控）
- `scripts/04_download_test_video.sh`：下载测试视频
- `scripts/05_infer_video_npu.sh`：视频单路推理
- `scripts/06_infer_video_multi_npu.sh`：视频三核并发推理
- `scripts/10_create_conda_env.sh`：创建 conda 环境

Python 程序：
- `scripts/infer_video_rknn.py`
- `scripts/infer_video_rknn_multi.py`

---

## 3. 快速开始

### 3.1 激活环境

```bash
eval "$(conda shell.bash hook)"
conda activate rk3588_yolov5
cd <workspace-root>/nanopc_t6_rk3588_yolov5_npu
```

### 3.2 安装 Python 依赖（requirements.txt）

```bash
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

说明：
- 若 `rknn-toolkit2` / `rknn-toolkit-lite2` 直接安装失败，请使用 Rockchip 官方提供的 wheel 文件安装。

首次部署可执行：

```bash
bash scripts/10_create_conda_env.sh
conda activate rk3588_yolov5
bash scripts/00_check_env.sh
```

### 3.3 模型准备

```bash
bash scripts/01_download_model.sh
bash scripts/02_convert_rknn.sh
```

默认模型路径：
- ONNX：`../rknn_model_zoo/examples/yolov5/model/yolov5m.onnx`
- RKNN：`output/yolov5_rk3588.rknn`

### 3.4 图片推理

```bash
bash scripts/03_infer_npu.sh
```

默认输入目录：
- `../rknn_model_zoo/examples/yolov5/model`

默认输出目录：
- `../rknn_model_zoo/examples/yolov5/python/result/`

提示：
- 若输入目录没有任何图片，先执行 `bash scripts/01_download_model.sh` 生成/准备测试图。

---

## 4. 视频推理

### 4.1 测试视频来源

```bash
# 可选：下载额外测试视频（保存到 output/test_video.mp4）
bash scripts/04_download_test_video.sh
```

默认视频路径：
- `test_assets/che.mp4`

### 4.2 单路视频推理

```bash
VIDEO_IN=test_assets/che.mp4 \
VIDEO_OUT=output/result_single.mp4 \
MAX_FRAMES=0 \
CORE_MASK=NPU_CORE_0_1_2 \
NPU_DEFAULT_FREQ=800000000 \
CROP_MODE=none \
CROP_RATIO=1.0 \
AUTO_TRANSCODE=1 \
bash scripts/05_infer_video_npu.sh
```

### 4.3 三核并发视频推理

```bash
VIDEO_IN=test_assets/che.mp4 \
VIDEO_OUT=output/result_multi.mp4 \
MAX_FRAMES=0 \
CORE_MASKS=NPU_CORE_0,NPU_CORE_1,NPU_CORE_2 \
NPU_DEFAULT_FREQ=800000000 \
MONITOR_INTERVAL=0.2 \
CROP_MODE=none \
CROP_RATIO=1.0 \
AUTO_TRANSCODE=1 \
bash scripts/06_infer_video_multi_npu.sh
```

---

## 5. 常用参数

- `MAX_FRAMES`：最大处理帧数，`0` 表示全视频
- `NPU_DEFAULT_FREQ`：NPU 固定频率（默认 `800000000`）
- `CORE_MASK`：单路模式核心掩码（默认 `NPU_CORE_0_1_2`）
- `CORE_MASKS`：并发模式核心掩码列表（默认 `NPU_CORE_0,NPU_CORE_1,NPU_CORE_2`）
- `MONITOR_INTERVAL`：NPU 负载采样间隔（秒）
- `CROP_MODE`：`none` 或 `center`
- `CROP_RATIO`：中心裁剪比例 `(0,1]`
- `AUTO_TRANSCODE`：`1` 开启自动转码兜底（依赖 `ffmpeg`）

---

## 6. 自动预处理

图片与视频帧在推理前会自动进行：

1. 通道规范化：灰度/BGRA -> BGR  
2. 数据类型规范化：非 `uint8` -> `uint8`  
3. 可选中心裁剪：`CROP_MODE=center` 且 `CROP_RATIO<1.0`  
4. 尺寸适配：letterbox 到模型输入（640）

支持图片后缀：
- `.jpg .jpeg .png .bmp .webp .tif .tiff`

---

## 7. 启动前模型检查

启动后会先 warmup 并验证：
- 输入是否为 `NHWC` 且 3 通道
- 输出是否为 3 路 4D 特征图
- 特征图尺寸是否匹配 `[20, 40, 80]`

检查通过会打印：
- `[INFO] startup check passed: model input/output shape/layout is valid`

---

## 8. 验收命令

```bash
# 图片
bash scripts/03_infer_npu.sh

# 单路视频（30 帧）
MAX_FRAMES=30 VIDEO_IN=test_assets/che.mp4 VIDEO_OUT=output/check_single.mp4 bash scripts/05_infer_video_npu.sh

# 三核并发视频（30 帧）
MAX_FRAMES=30 VIDEO_IN=test_assets/che.mp4 VIDEO_OUT=output/check_multi.mp4 bash scripts/06_infer_video_multi_npu.sh
```

---

## 9. 关键输出文件

- RKNN 模型：`output/yolov5_rk3588.rknn`
- 图片结果：`../rknn_model_zoo/examples/yolov5/python/result/bus.jpg`
- 单路视频结果：`output/result_single.mp4`
- 三核并发视频结果：`output/result_multi.mp4`
- 图片推理 NPU 日志：`output/npu_load.log`
- 并发推理 NPU 日志：`output/multi_rknpu_load.log`
