# Wan 模型输入输出与核心结构对照

下面是按 **input -> 核心处理 -> output** 维度整理的对照表。

| 维度 | 通用 I2V | Animate（表情/人脸驱动） | S2V（语音驱动） |
|---|---|---|---|
| 主要入口 | `wan/image2video.py` | `wan/animate.py` | `wan/modules/s2v/model_s2v.py`（由对应 pipeline 调用） |
| 主干模型类 | `WanModel` in `wan/modules/model.py` | `WanAnimateModel` in `wan/modules/animate/model_animate.py` | `WanModel_S2V` in `wan/modules/s2v/model_s2v.py` |
| 主要输入 | 文本 + 参考图像（首帧）+ 噪声 latent | 文本 + 参考图 + 姿态视频 + 人脸视频 +（可选背景/掩码）+ 噪声 latent | 文本 + 参考/运动 latent + 条件帧（pose）+ 音频特征 + 噪声 latent |
| 文本编码 | `T5EncoderModel` | `T5EncoderModel` | `T5EncoderModel` |
| VAE 编解码 | `Wan2_1_VAE` | `Wan2_1_VAE` | 同样走 VAE latent 空间 |
| Transformer 主干 | 标准 `WanAttentionBlock` 堆叠 | `WanAnimateAttentionBlock`（改造版） | `WanS2VAttentionBlock`（改造版） |
| 任务特化模块 | 基本无（偏通用） | `FaceEncoder` + `FaceAdapter` + `motion_encoder` + CLIP 图像特征注入 | `CausalAudioEncoder` + `AudioInjector_WAN` + `Motioner/FramePack` |
| 特化注入位置 | 主要是通用 cross-attn | `after_patch_embedding`（pose/face 注入）+ `after_transformer_block`（face adapter 融合） | `inject_motion` + `after_transformer_block`（audio 注入） |
| 采样器流程 | UniPC / DPM++ 去噪循环 | UniPC / DPM++ 去噪循环（多了时序切片与参考拼接） | 同类扩散去噪流程 |
| 输出 | 生成视频帧 | 生成带表情/动作控制的视频帧 | 生成口型/语音同步导向视频帧 |
| 与其它模型“是否同核心” | 与其它共享扩散骨架 | 骨架相同，但控制分支差异大 | 骨架相同，但音频/运动分支差异大 |

## 一句话总结

- 共用的是“扩散主干范式”（patch/time/text/blocks/head/unpatchify）。
- 差异最大的是“控制信号如何编码并注入”（人脸、姿态、音频、运动 token）。
