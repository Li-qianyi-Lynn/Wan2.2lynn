# Wan2.2 代码仓结构说明

## 1. 仓库概览

基于 [Wan2.2](https://github.com/Wan-Video/Wan2.2) 的视频生成框架，支持多种任务：T2V、I2V、TI2V、S2V、Animate（角色动画与替换）。入口为根目录 `generate.py`，按 `--task` 选择配置与对应 pipeline。

## 2. 顶层结构

```
Wan2.2lynn/
├── generate.py              # 统一入口：解析参数、选择 config、调用对应 pipeline
├── README.md                # 项目说明与运行示例
├── requirements.txt         # 基础依赖
├── requirements_s2v.txt    # S2V 额外依赖（如 CosyVoice）
├── requirements_animate.txt # Animate 相关依赖
├── pyproject.toml / Makefile
├── wan/                     # 主代码包
├── tests/                   # 测试脚本
├── examples/                # 示例输入（如 wan_animate 的 video/image）
└── assets/                  # 静态资源（如 logo）
```

## 3. 配置与任务映射

- **wan/configs/__init__.py**  
  从各子配置汇总 `WAN_CONFIGS`（t2v-A14B, i2v-A14B, ti2v-5B, s2v-14B, animate-14B）、`SIZE_CONFIGS`、`SUPPORTED_SIZES`。
- **wan/configs/shared_config.py**  
  公共配置（如 VAE、noise schedule、dtype）。
- **wan/configs/wan_*.py**  
  - `wan_t2v_A14B.py`：纯文生视频  
  - `wan_i2v_A14B.py`：图生视频  
  - `wan_ti2v_5B.py`：文+图生视频（5B，720P）  
  - `wan_s2v_14B.py`：语音/音频生视频  
  - `wan_animate_14B.py`：角色动画与替换（含 `use_face_encoder`、`motion_encoder_dim`、`lora_checkpoint` 等）

## 4. 主包 `wan/` 结构

### 4.1 入口与 pipeline

- **wan/__init__.py**：包入口。
- **wan/text2video.py**：T2V pipeline（加载 T5、DiT、VAE，按 prompt 生成）。
- **wan/image2video.py**：I2V pipeline（首帧 + prompt）。
- **wan/textimage2video.py**：TI2V pipeline（可选图+文）。
- **wan/speech2video.py**：S2V pipeline（图 + 音频 + 可选 pose；加载 S2V 专用 DiT、音频编码器、可选 motioner）。
- **wan/animate.py**：Animate pipeline（`WanAnimate` 类：加载 T5、CLIP、VAE、WanAnimateModel；读取预处理后的 pose/face/ref；支持 animation / replacement，可选 Relighting LoRA）。

### 4.2 核心模块 `wan/modules/`

- **model.py**  
  基础 DiT 组件：WanModel、WanAttentionBlock、WanSelfAttention、Head、RoPE、sinusoidal embedding 等，被 T2V/I2V/TI2V 及部分 Animate/S2V 复用。
- **t5.py**  
  文本编码器（T5），供所有需要文本条件的任务使用。
- **vae2_1.py / vae2_2.py**  
  视频 VAE（2.1 用于 Animate，2.2 用于部分高分辨率/新模型）。
- **tokenizers.py**  
  文本 tokenizer 封装。

### 4.3 Animate 专用（`wan/modules/animate/`）

- **model_animate.py**  
  - WanAnimateModel：patch_embedding、pose_patch_embedding、text_embedding、time_embedding；  
  - 条件融合：`after_patch_embedding()` 中 pose latents 与主 patch 合并，face → motion_encoder → face_encoder → motion_vec；  
  - WanAnimateAttentionBlock：self-attn + cross-attn（context 含 CLIP 图像前 257 token）；  
  - FaceAdapter 在部分 block 注入 face/motion；HeadAnimate 输出。
- **face_blocks.py**  
  FaceEncoder、FaceAdapter（将 face 隐式特征注入到 transformer）。
- **motion_encoder.py**  
  基于 LIA 的 Generator：从 face 图像提取 style（identity/personality）与 motion（20 维 latent → 512 维 direction），供 `get_motion()` 在推理中使用。
- **animate_utils.py**  
  `extract_personality_and_motion()`（供 PCA/分析）、`get_loraconfig()`（Relighting LoRA）、TensorList 等。
- **clip.py / xlm_roberta.py**  
  CLIP 视觉编码与 XLM-Roberta，用于参考图 → context。
- **extract_animate_features.py**  
  从 face 视频/图像提取 personality 与 motion 特征并保存 .npy，供 pca_analysis 等使用。
- **pca_analysis.py**  
  对 personality/motion 做 PCA 分析（与主推理管线解耦）。

### 4.4 Animate 预处理（`wan/modules/animate/preprocess/`）

- **preprocess_data.py**  
  主预处理入口：根据 animation / replacement 模式调用不同 pipeline，输出 `src_pose.mp4`、`src_face.mp4`、`src_ref.png` 等。
- **process_pipepline.py**  
  预处理流程编排。
- **pose2d.py / pose2d_utils.py**  
  2D 姿态估计与工具。
- **retarget_pose.py**  
  姿态重定向（animation 时对齐到参考角色）。
- **video_predictor.py**  
  视频级预测相关。
- **sam_utils.py**  
  SAM 相关工具（replacement 时分割等）。
- **human_visualization.py**  
  人体/骨架可视化。
- **utils.py**  
  预处理通用工具。
- **UserGuider.md**  
  预处理使用说明。

### 4.5 S2V 专用（`wan/modules/s2v/`）

- **model_s2v.py**  
  S2V 用 DiT：ref_latents（参考图）、cond_states（可选 pose 与主 latent 相加）、context（文本）、CausalAudioEncoder + AudioInjector 在指定层注入音频；无独立 skeleton 或 face 编码器。
- **audio_encoder.py**  
  音频编码（如 wav2vec 类）与时间对齐。
- **audio_utils.py**  
  AudioInjector_WAN、CausalAudioEncoder 等。
- **motioner.py**  
  FramePackMotioner、MotionerTransformers，处理可选 pose 视频。
- **s2v_utils.py**  
  RoPE 预计算等工具。
- **auxi_blocks.py**  
  S2V 辅助网络块。

## 5. 工具与分布式

- **wan/utils/**  
  - `fm_solvers.py` / `fm_solvers_unipc.py`：采样器（如 Flow DPM++、UniPC）。  
  - `prompt_extend.py`：Prompt 扩展（DashScope / 本地 Qwen）。  
  - `utils.py`：保存视频、合并音视频等。  
  - `system_prompt.py`：扩展用系统提示。  
  - `qwen_vl_utils.py`：Qwen-VL 相关工具。
- **wan/distributed/**  
  - `fsdp.py`：FSDP 分片。  
  - `sequence_parallel.py`：序列并行（含 attention 重写）。  
  - `ulysses.py`：DeepSpeed Ulysses。  
  - `util.py`：分布式初始化与 world size 等。

## 6. 数据流小结（与网站描述对应）

- **Animate-14B**  
  - 参考图 + 时间帧引导 + 环境信息：通过统一输入符号（conditioning_pixel_values、face_pixel_values、refer_pixel_values、replace 时的 bg/mask）进入 pipeline。  
  - 身体动作：skeleton → pose 视频 → VAE → pose_patch_embedding，与主 patch 空间对齐合并。  
  - 表情：face 图像序列 → motion_encoder 隐式特征 → face_encoder → FaceAdapter 注入。  
  - 角色替换：Relighting LoRA + 背景/遮罩输入。
- **S2V-14B**  
  身份与动作由「参考图 + 文本 + 音频 + 可选 pose」共同决定，无独立 skeleton/face 编码器。
- **T2V / I2V / TI2V**  
  无显式动作或 identity 编码器，仅文本（及 I2V/TI2V 的参考图）条件。


各任务对动作（motion）和身份/外观（identity）的建模方式不同，对应项目网站 Wan-Animate 里说的「skeleton + 隐式面部特征 + 统一符号表示」主要针对 Wan2.2-Animate，其它模型是另一套设计。

“skeleton signals that are merged via spatial alignment” → 仅在 Animate-14B 中实现：pose 视频经 VAE 得到 pose latents，再通过 pose_patch_embedding 与主 patch 在空间上合并（见 model_animate.py 的 after_patch_embedding 里对 pose_latents 的加性合并）。

“implicit features extracted from face images as the driving signal” → 也仅在 Animate-14B：face_pixel_values → motion_encoder（LIA 风格 Encoder，输出 20 维 motion + 512 维 style）→ face_encoder → 在部分 transformer block 用 FaceAdapter 注入。

“Relighting LoRA” → 仅 Animate-14B 的 replacement 模式 使用（use_relighting_lora），用于角色与新环境的光照融合。

## 7. Animate 与 Replacement 模式区别

Wan-Animate 支持两种用法：**Animation（动画）** 与 **Replacement（角色替换）**，从输入目标、预处理产物到推理时的条件都不同。

| 维度 | Animation 模式 | Replacement 模式 |
|------|----------------|------------------|
| **目标** | 用「驱动视频」里的动作和表情，驱动「角色参考图」生成一段**全新视频**（背景可任意，由模型生成）。 | 把原视频里的角色**替换**成「角色参考图」中的形象，保留原视频的**背景、光照、构图**，只换人。 |
| **预处理** | `--retarget_flag` 可选（姿态重定向，使驱动 pose 对齐到参考图体型）；可选 `--use_flux` 做首帧编辑。**不**加载 SAM2，**不**生成背景/遮罩。 | `--replace_flag`。会加载 **SAM2** 做视频分割，得到人物 mask；生成 **src_bg.mp4**（去掉人后的背景）、**src_mask.mp4**（人物遮罩）。无 retarget。 |
| **预处理输出** | `src_pose.mp4`、`src_face.mp4`、`src_ref.png`（三者共用）。 | 上述三者 **+** `src_bg.mp4`、`src_mask.mp4`。 |
| **推理输入** | 仅 pose / face / ref；时间引导帧 `refer_t_pixel_values` 为上一 clip 的生成帧或零。 | 额外读入 `bg_pixel_values`、`mask_pixel_values`；首帧/引导帧的 VAE 条件会 **concat 原视频背景**（`y_reft` 里用 `bg_pixel_values`），并用 mask 指定「哪里是角色、哪里是背景」。 |
| **Relighting LoRA** | 不使用。 | **使用**（`use_relighting_lora=True` 时加载 `relighting_lora.ckpt`），让生成角色与原场景光照、色调一致，实现无缝融合。 |

**小结**：Animation 是「角色 + 动作/表情 → 新视频」；Replacement 是「角色 + 原视频背景/遮罩 → 原视频里换人」，并依赖 Relighting LoRA 做环境融合。

