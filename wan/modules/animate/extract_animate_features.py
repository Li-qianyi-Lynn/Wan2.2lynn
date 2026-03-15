# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
"""
从 face 图像中提取 personality 与 motion 特征并保存为 .npy，供 pca_analysis.py 使用。

图片从哪里来？
  若用预处理流程，face 来自 process_results 里的 src_face.mp4，可直接用 --face_video 指定该视频。

在项目根目录下运行（推荐，避免与已安装的 wan 冲突）：
  python wan/modules/animate/extract_animate_features.py \\
    --checkpoint_dir /path/to/wan_animate_ckpt \\
    --face_video examples/wan_animate/animate/process_results/src_face.mp4 \\
    --out_dir ./features

或使用 PYTHONPATH 再 -m：
  PYTHONPATH=/path/to/Wan2.2lynn python -m wan.modules.animate.extract_animate_features ...

用法二（已有 face 的 .npy）：同上，把 --face_video 换成 face_npy 路径。
用法三（从图片目录读取）：同上，把 --face_video 换成 --image_dir /path/to/face_images。

face_npy 格式：numpy 数组 shape (N, 3, 512, 512)，值域 [-1, 1]。
"""

import argparse
import os
import sys
import numpy as np
import torch

# 保证从项目根能找到本项目的 wan 包（避免 pip 安装的 wan 覆盖）
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from wan.modules.animate.motion_encoder import Generator
from wan.modules.animate.animate_utils import extract_personality_and_motion


def _load_state_dict_from_checkpoint(checkpoint_dir):
    """从 checkpoint 目录加载完整 state_dict；支持多分片 .safetensors 与 pytorch_model.bin。"""
    import glob
    state = {}
    # 加载所有 .safetensors 分片并合并（Wan2.2-Animate-14B 可能有多片）
    st_files = sorted(glob.glob(os.path.join(checkpoint_dir, "*.safetensors")))
    if st_files:
        try:
            from safetensors import safe_open
            for path in st_files:
                with safe_open(path, framework="pt", device="cpu") as f:
                    for k in f.keys():
                        if k in state:
                            continue
                        state[k] = f.get_tensor(k)
            if state:
                return state
        except Exception as e:
            print(f"Warning: loading safetensors failed: {e}")
    bin_path = os.path.join(checkpoint_dir, "pytorch_model.bin")
    if os.path.isfile(bin_path):
        return torch.load(bin_path, map_location="cpu", weights_only=True)
    raise FileNotFoundError(
        f"在 {checkpoint_dir} 下未找到 .safetensors 或 pytorch_model.bin"
    )


def load_motion_encoder_only(checkpoint_dir, device="cuda"):
    """只加载 motion_encoder 权重，不加载完整 14B 模型。支持多种 checkpoint 的 key 前缀。"""
    full_state = _load_state_dict_from_checkpoint(checkpoint_dir)
    # 常见前缀：diffusers 直接存为 motion_encoder.；有的仓库存为 model. 或 noise_model.
    for prefix in ("motion_encoder.", "model.motion_encoder.", "noise_model.motion_encoder."):
        subset = {k[len(prefix):]: v for k, v in full_state.items() if k.startswith(prefix)}
        if subset:
            model = Generator(size=512, style_dim=512, motion_dim=20)
            model.load_state_dict(subset, strict=True)
            model = model.to(device).eval()
            return model
    # 未找到时打印部分 key 便于排查
    sample = [k for k in list(full_state.keys())[:30] if "motion" in k.lower() or "enc" in k.lower()]
    if not sample:
        sample = list(full_state.keys())[:15]
    raise KeyError(
        f"state_dict 中未找到 motion_encoder 相关 key。"
        f" 示例 key: {sample}"
    )


def load_face_images_from_dir(image_dir, size=512):
    """从目录读取图片，resize 并归一化到 [-1,1]，返回 (N, 3, size, size)。"""
    try:
        from PIL import Image
    except ImportError:
        raise ImportError("请安装 Pillow: pip install Pillow")
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    paths = []
    for name in sorted(os.listdir(image_dir)):
        if name.lower().endswith(exts):
            paths.append(os.path.join(image_dir, name))
    if not paths:
        raise FileNotFoundError(f"在 {image_dir} 下未找到图片")
    arrs = []
    for p in paths:
        img = Image.open(p).convert("RGB")
        img = np.array(img).astype(np.float32) / 127.5 - 1.0  # [0,255] -> [-1,1]
        # HWC -> CHW, then resize to (size, size)
        img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)  # 1,3,H,W
        img = torch.nn.functional.interpolate(
            img, size=(size, size), mode="bilinear", align_corners=False
        )
        arrs.append(img)
    out = torch.cat(arrs, dim=0)  # N,3,H,W
    return out


def load_face_images_from_video(video_path, size=512):
    """
    从 face 视频（如预处理得到的 src_face.mp4）按帧读取，resize 并归一化到 [-1,1]。
    返回 (N, 3, size, size)，N 为帧数。
    """
    try:
        from decord import VideoReader
        from decord import cpu
    except ImportError:
        try:
            import cv2
        except ImportError:
            raise ImportError("请安装 decord 或 opencv-python: pip install decord 或 pip install opencv-python")
        # fallback: cv2
        cap = cv2.VideoCapture(video_path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        cap.release()
        if not frames:
            raise ValueError(f"视频中未读到帧: {video_path}")
        arr = np.stack(frames).astype(np.float32) / 127.5 - 1.0  # (T,H,W,3)
        arr = torch.from_numpy(arr).permute(0, 3, 1, 2)  # (T,3,H,W)
        arr = torch.nn.functional.interpolate(
            arr, size=(size, size), mode="bilinear", align_corners=False
        )
        return arr
    vr = VideoReader(video_path, ctx=cpu(0))
    n = len(vr)
    # get_batch 返回 (T, H, W, C) uint8
    frames = vr.get_batch(list(range(n))).asnumpy()  # (T, H, W, 3)
    arr = torch.from_numpy(frames).float().permute(0, 3, 1, 2) / 127.5 - 1.0  # (T, 3, H, W)
    if arr.shape[2] != size or arr.shape[3] != size:
        arr = torch.nn.functional.interpolate(
            arr, size=(size, size), mode="bilinear", align_corners=False
        )
    return arr


def main():
    parser = argparse.ArgumentParser(description="提取 personality / motion 特征并保存 .npy")
    parser.add_argument("--checkpoint_dir", type=str, required=True, help="WanAnimate 权重目录")
    parser.add_argument(
        "face_npy",
        type=str,
        nargs="?",
        default=None,
        help="face 数组 .npy，shape (N, 3, 512, 512)，值域 [-1, 1]",
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        default=None,
        help="或：face 图片目录，将按文件名排序并 resize 到 512x512",
    )
    parser.add_argument(
        "--face_video",
        type=str,
        default=None,
        help="或：face 视频路径，如 process_results/src_face.mp4，会按帧提取并 resize 到 512x512",
    )
    parser.add_argument("--out_dir", type=str, default="./features", help="输出目录")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    if args.face_video:
        faces = load_face_images_from_video(args.face_video)
    elif args.face_npy:
        faces = np.load(args.face_npy)
        if isinstance(faces, np.ndarray):
            faces = torch.from_numpy(faces).float()
        else:
            faces = torch.tensor(faces).float()
    elif args.image_dir:
        faces = load_face_images_from_dir(args.image_dir)
    else:
        raise ValueError("请指定 --face_video、face_npy 或 --image_dir")

    if faces.dim() == 3:
        faces = faces.unsqueeze(0)
    # 确保 (N, 3, 512, 512)
    if faces.shape[2] != 512 or faces.shape[3] != 512:
        faces = torch.nn.functional.interpolate(
            faces, size=(512, 512), mode="bilinear", align_corners=False
        )
    N = faces.shape[0]
    print(f"Face 数量: {N}, shape: {tuple(faces.shape)}")

    motion_encoder = load_motion_encoder_only(args.checkpoint_dir, device=args.device)
    personality, motion_20, motion_512 = extract_personality_and_motion(
        motion_encoder, faces.to(args.device), batch_size=args.batch_size
    )

    os.makedirs(args.out_dir, exist_ok=True)
    np.save(os.path.join(args.out_dir, "personality.npy"), personality.numpy())
    np.save(os.path.join(args.out_dir, "motion_20.npy"), motion_20.numpy())
    np.save(os.path.join(args.out_dir, "motion_512.npy"), motion_512.numpy())
    print(f"已保存到 {args.out_dir}: personality.npy ({personality.shape}), motion_20.npy ({motion_20.shape}), motion_512.npy ({motion_512.shape})")
    p_npy = os.path.join(args.out_dir, "personality.npy")
    m_npy = os.path.join(args.out_dir, "motion_20.npy")
    print(f"下一步: python -m wan.modules.animate.pca_analysis {p_npy} {m_npy} --out-dir ./pca_outputs")


if __name__ == "__main__":
    main()
