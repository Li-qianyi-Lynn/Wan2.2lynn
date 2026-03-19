import argparse
import os

import wan
from wan.configs import WAN_CONFIGS

def parse_args():
    parser = argparse.ArgumentParser("Wan Animate Image Edit")
    parser.add_argument("--ckpt_dir", type=str, required=True,
                        help="Wan-A14B 动画模型的 checkpoint 目录")
    parser.add_argument("--task", type=str, default="animate-14B",
                        choices=WAN_CONFIGS.keys(),
                        help="使用的配置名，默认 animate-14B")
    parser.add_argument("--img", type=str, required=True,
                        help="输入人脸 jpg 路径")
    parser.add_argument("--out", type=str, default="output.jpg",
                        help="输出图片保存路径")
    parser.add_argument("--prompt", type=str, required=True,
                        help="正向 prompt，例如：'the same person, smiling, fair skin'")
    parser.add_argument("--n_prompt", type=str, default="",
                        help="负向 prompt，不填则使用默认")
    parser.add_argument("--guide_scale", type=float, default=3.0,
                        help="文本控制强度，>1 有效")
    parser.add_argument("--steps", type=int, default=20,
                        help="采样步数")
    parser.add_argument("--shift", type=float, default=5.0,
                        help="噪声调度 shift，建议跟训练配置一致")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子，<0 则使用随机种子")
    parser.add_argument("--device_id", type=int, default=0,
                        help="GPU id")
    parser.add_argument("--t5_cpu", action="store_true",
                        help="把 T5 放在 CPU 上（显存紧张时用）")
    parser.add_argument("--offload_model", action="store_true",
                        help="推理中把部分模块挪回 CPU 以省显存")
    return parser.parse_args()

def main():
    args = parse_args()

    cfg = WAN_CONFIGS[args.task]

    # 初始化 WanAnimate
    wan_animate = wan.WanAnimate(
        config=cfg,
        checkpoint_dir=args.ckpt_dir,
        device_id=args.device_id,
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_sp=False,
        t5_cpu=args.t5_cpu,
        init_on_cpu=True,
        convert_model_dtype=False,
        use_relighting_lora=False,
    )

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    _ = wan_animate.generate_image(
        img_path=args.img,
        input_prompt=args.prompt,
        n_prompt=args.n_prompt,
        guide_scale=args.guide_scale,
        sampling_steps=args.steps,
        shift=args.shift,
        seed=args.seed,
        offload_model=args.offload_model,
        save_path=args.out,
    )

    print(f"[WanAnimate] Saved edited image to: {args.out}")

if __name__ == "__main__":
    main()