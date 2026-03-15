# Personality 与 Motion 的 PCA 分析（汇报用）

## 1. 背景：模型里 personality 和 motion 是什么？

- **Personality（身份/风格）**：来自 motion encoder 的 `enc_app`，512 维向量，表示“这个人长什么样、什么风格”。同一人不同帧应尽量接近。
- **Motion（动作）**：来自 `enc_motion`，20 维 latent，表示“这一帧的表情/动作”；再通过 `direction` 映射成 512 维注入到主模型。

把这两路**分开做 PCA**，可以看：
- 身份是否在 512 维里聚类（同一人聚在一起）；
- 动作是否在 20 维里有明显结构（前几维就能解释大部分方差）。

## 2. 图片从哪里来？

- **预处理得到的 face 视频**：`examples/wan_animate/animate/process_results/src_face.mp4`  
  预处理流程会生成 `src_face.mp4`（每帧是一张裁剪对齐的人脸），可直接用 `--face_video` 指定，脚本会按帧提取。
- 或：已有 face 的 .npy、或一目录的 face 图片（见下方命令）。

## 3. WanAnimate 权重的实际路径怎么找？

`--checkpoint_dir` 就是**你跑 animate 生成时用的那个权重目录**，和 `generate.py --ckpt_dir` 是同一个路径。

- **若你已经在集群上跑通过 animate**：看当时用的 `--ckpt_dir` 或脚本里的 `checkpoint_dir` 变量，那就是。例如：
  ```bash
  python generate.py --task animate-14B --ckpt_dir /scratch/li.qianyi/models/wan_animate ...
  ```
  这里 `--checkpoint_dir` 就填 `/scratch/li.qianyi/models/wan_animate`（按你实际路径改）。

- **目录里通常会有**：主模型权重（如 `pytorch_model.bin` 或 `*.safetensors`）、以及配置等。T5/CLIP/VAE 的路径是在这个目录下按 config 里的文件名找的（如 `models_t5_umt5-xxl-enc-bf16.pth`）。提取脚本只读**主模型**那部分（里面有 `motion_encoder`），所以只要该目录下存在 `.safetensors` 或 `pytorch_model.bin` 即可。

- **不确定时**：在该目录下执行 `ls`，看是否有 `pytorch_model.bin` 或 `*.safetensors`；若没有，可能是把主模型放在子目录里，需要把子目录路径作为 `--checkpoint_dir` 再试。

## 4. 流程（两步）

### 步骤一：提取特征

在**有 WanAnimate checkpoint 的环境**（如集群）运行：

在**项目根目录**下执行（避免 No module named wan.modules.animate.extract_animate_features）：

```bash
# 方式 A（推荐）：直接用预处理得到的 src_face.mp4
# 把 /path/to/wan_animate_ckpt 换成上面「权重的实际路径」
python wan/modules/animate/extract_animate_features.py \
  --checkpoint_dir /path/to/wan_animate_ckpt \
  --face_video examples/wan_animate/animate/process_results/src_face.mp4 \
  --out_dir ./features

# 若报错 No module named wan...，可改用：PYTHONPATH=$PWD python -m wan.modules.animate.extract_animate_features ...

# 方式 B：已有 face 的 .npy
python wan/modules/animate/extract_animate_features.py \
  --checkpoint_dir /path/to/wan_animate_ckpt \
  /path/to/faces.npy \
  --out_dir ./features

# 方式 C：从 face 图片目录
python wan/modules/animate/extract_animate_features.py \
  --checkpoint_dir /path/to/wan_animate_ckpt \
  --image_dir /path/to/face_images \
  --out_dir ./features
```

会得到：`features/personality.npy`、`features/motion_20.npy`、`features/motion_512.npy`。

### 步骤二：做 PCA 并出图

在**本机或任意有 numpy/matplotlib/sklearn 的环境**运行（不需要 GPU/checkpoint）：

```bash
python -m wan.modules.animate.pca_analysis \
  features/personality.npy \
  features/motion_20.npy \
  --out-dir ./pca_outputs
```

可选：`--labels labels.npy`（与样本一一对应的 ID，用于散点图按人/视频着色）。

会得到：

- `pca_outputs/personality_pca.png`：Personality 的方差解释率 + PC1 vs PC2 散点图  
- `pca_outputs/motion_pca.png`：Motion 的方差解释率 + PC1 vs PC2 散点图  
- `pca_outputs/report_summary.txt`：文字版汇报要点  

## 5. 统计与汇报时怎么说（给老师汇报用）

### 5.1 PCA 在做什么（一句话）

PCA 把高维特征（512 或 20 维）投影到少数几个“主成分”上，使投影后方差最大，用来看**主要变化方向**和**降维后能保留多少信息**。

### 5.2 汇报时可以说的点

1. **数据**  
   “我们用了 N 帧 face 图像，每帧提取 512 维 personality 和 20 维 motion，并做了标准化后再做 PCA。”

2. **Personality**  
   - “前 2 个主成分的累计方差是 XX%，说明身份/风格在低维空间里比较集中。”  
   - 若做了按人着色：“同一人在 PC1–PC2 平面上聚在一起，说明模型的身份表示具有区分度。”

3. **Motion**  
   - “Motion 只有 20 维，前 2 维就解释了 XX% 的方差，说明动作变化主要集中在少数方向上。”  
   - “不同动作在 PC1–PC2 上分布不同，说明 20 维 motion 空间能区分不同动作模式。”

4. **结论**  
   “模型把 face 解耦成 personality（身份）和 motion（动作）两路；对两路分别做 PCA 表明身份表示紧凑、可聚类，动作表示在低维上也有清晰结构，便于后续控制或分析。”

### 5.3 若老师问“为什么分开做 PCA？”

- Personality 和 motion 语义不同（谁 vs 怎么动），分开做可以看到**各自**的主要变化方向。  
- 若混在一起做，身份和动作混在同一套主成分里，不便于解释。

## 6. 依赖

- 特征提取：`torch`，以及项目依赖；可选 `safetensors`（若 checkpoint 为 .safetensors）。  
- PCA 脚本：`numpy`、`scikit-learn`、`matplotlib`（若不需要出图可去掉画图部分）。

```bash
pip install numpy matplotlib scikit-learn
```

## 7. 在推理流程里直接保存特征（可选）

若你已经在跑 animate 推理且已有 `face_pixel_values`（例如 shape `(1, 3, T, 512, 512)`），可以在推理代码里对当前 batch 做一次提取并保存，无需再跑 `extract_animate_features`：

```python
from einops import rearrange
from wan.modules.animate.animate_utils import extract_personality_and_motion

# face_pixel_values: (1, 3, T, 512, 512)
face_flat = rearrange(face_pixel_values[0], "c t h w -> t c h w")
personality, motion_20, motion_512 = extract_personality_and_motion(
    model.noise_model.motion_encoder, face_flat, batch_size=8
)
import numpy as np
np.save("personality.npy", personality.cpu().numpy())
np.save("motion_20.npy", motion_20.cpu().numpy())
```

再用 `pca_analysis` 读这两个 `.npy` 即可。
