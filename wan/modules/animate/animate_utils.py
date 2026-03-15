# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import torch
import numbers
from peft import LoraConfig


def extract_personality_and_motion(motion_encoder, face_images, batch_size=8):
    """
    从 face 图像中分别提取 personality（身份/风格）和 motion（动作）特征，用于后续 PCA 等统计分析。

    Personality：来自 Encoder 的 enc_app，512 维，同一人不同帧应较稳定。
    Motion：来自 Encoder 的 enc_motion，20 维原始动作码；可选再通过 direction 得到 512 维 motion 向量。

    Args:
        motion_encoder: wan.modules.animate.motion_encoder.Generator 实例（即 model.motion_encoder）
        face_images: torch.Tensor, shape (N, 3, H, W)，已归一化到 [-1, 1]，如 512x512
        batch_size: 前向时每批帧数，避免 OOM

    Returns:
        personality: (N, 512) float32，身份/风格特征
        motion_20: (N, 20) float32，原始动作 latent
        motion_512: (N, 512) float32，通过 direction 后的 motion 向量（与模型内实际注入的一致）
    """
    motion_encoder.eval()
    N = face_images.shape[0]
    device = next(motion_encoder.parameters()).device
    face_images = face_images.to(device=device, dtype=torch.float32)

    personality_list, motion_20_list, motion_512_list = [], [], []
    with torch.no_grad():
        for i in range(0, N, batch_size):
            batch = face_images[i : i + batch_size]
            # personality: 512-d style/appearance
            p = motion_encoder.enc.enc_app(batch)
            personality_list.append(p.cpu().float())
            # motion: 20-d latent
            m20 = motion_encoder.enc.enc_motion(batch)
            motion_20_list.append(m20.cpu().float())
            # motion 512-d (direction 输出，与 get_motion 一致)
            with torch.cuda.amp.autocast(dtype=torch.float32):
                m512 = motion_encoder.dec.direction(m20)
            motion_512_list.append(m512.cpu().float())

    personality = torch.cat(personality_list, dim=0)
    motion_20 = torch.cat(motion_20_list, dim=0)
    motion_512 = torch.cat(motion_512_list, dim=0)
    return personality, motion_20, motion_512


def get_loraconfig(transformer, rank=128, alpha=128, init_lora_weights="gaussian"):
    target_modules = []
    for name, module in transformer.named_modules():
        if "blocks" in name and "face" not in name and "modulation" not in name and isinstance(module, torch.nn.Linear):
            target_modules.append(name)

    transformer_lora_config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        init_lora_weights=init_lora_weights,
        target_modules=target_modules,
    )
    return transformer_lora_config



class TensorList(object):

    def __init__(self, tensors):
        """
        tensors: a list of torch.Tensor objects. No need to have uniform shape.
        """
        assert isinstance(tensors, (list, tuple))
        assert all(isinstance(u, torch.Tensor) for u in tensors)
        assert len(set([u.ndim for u in tensors])) == 1
        assert len(set([u.dtype for u in tensors])) == 1
        assert len(set([u.device for u in tensors])) == 1
        self.tensors = tensors
    
    def to(self, *args, **kwargs):
        return TensorList([u.to(*args, **kwargs) for u in self.tensors])
    
    def size(self, dim):
        assert dim == 0, 'only support get the 0th size'
        return len(self.tensors)
    
    def pow(self, *args, **kwargs):
        return TensorList([u.pow(*args, **kwargs) for u in self.tensors])
    
    def squeeze(self, dim):
        assert dim != 0
        if dim > 0:
            dim -= 1
        return TensorList([u.squeeze(dim) for u in self.tensors])
    
    def type(self, *args, **kwargs):
        return TensorList([u.type(*args, **kwargs) for u in self.tensors])
    
    def type_as(self, other):
        assert isinstance(other, (torch.Tensor, TensorList))
        if isinstance(other, torch.Tensor):
            return TensorList([u.type_as(other) for u in self.tensors])
        else:
            return TensorList([u.type(other.dtype) for u in self.tensors])
    
    @property
    def dtype(self):
        return self.tensors[0].dtype
    
    @property
    def device(self):
        return self.tensors[0].device
    
    @property
    def ndim(self):
        return 1 + self.tensors[0].ndim
    
    def __getitem__(self, index):
        return self.tensors[index]
    
    def __len__(self):
        return len(self.tensors)
    
    def __add__(self, other):
        return self._apply(other, lambda u, v: u + v)
    
    def __radd__(self, other):
        return self._apply(other, lambda u, v: v + u)
    
    def __sub__(self, other):
        return self._apply(other, lambda u, v: u - v)
    
    def __rsub__(self, other):
        return self._apply(other, lambda u, v: v - u)
    
    def __mul__(self, other):
        return self._apply(other, lambda u, v: u * v)
    
    def __rmul__(self, other):
        return self._apply(other, lambda u, v: v * u)
    
    def __floordiv__(self, other):
        return self._apply(other, lambda u, v: u // v)
    
    def __truediv__(self, other):
        return self._apply(other, lambda u, v: u / v)
    
    def __rfloordiv__(self, other):
        return self._apply(other, lambda u, v: v // u)
    
    def __rtruediv__(self, other):
        return self._apply(other, lambda u, v: v / u)
    
    def __pow__(self, other):
        return self._apply(other, lambda u, v: u ** v)
    
    def __rpow__(self, other):
        return self._apply(other, lambda u, v: v ** u)
    
    def __neg__(self):
        return TensorList([-u for u in self.tensors])
    
    def __iter__(self):
        for tensor in self.tensors:
            yield tensor
    
    def __repr__(self):
        return 'TensorList: \n' + repr(self.tensors)

    def _apply(self, other, op):
        if isinstance(other, (list, tuple, TensorList)) or (
            isinstance(other, torch.Tensor) and (
                other.numel() > 1 or other.ndim > 1
            )
        ):
            assert len(other) == len(self.tensors)
            return TensorList([op(u, v) for u, v in zip(self.tensors, other)])
        elif isinstance(other, numbers.Number) or (
            isinstance(other, torch.Tensor) and (
                other.numel() == 1 and other.ndim <= 1
            )
        ):
            return TensorList([op(u, other) for u in self.tensors])
        else:
            raise TypeError(
                f'unsupported operand for *: "TensorList" and "{type(other)}"'
            )