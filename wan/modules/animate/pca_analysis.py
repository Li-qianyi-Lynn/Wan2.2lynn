# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
"""
Personality 与 Motion 的 PCA 分析脚本（面向汇报与统计理解）

使用步骤：
  1. 用 animate_utils.extract_personality_and_motion 从视频/图像提取特征，保存为 .npy
  2. 本脚本读取 .npy，分别对 personality 和 motion 做 PCA，出图、出表、出汇报要点

依赖：pip install numpy matplotlib scikit-learn
"""

import argparse
import os
import numpy as np

try:
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
except ImportError:
    raise ImportError("请安装: pip install scikit-learn")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


def _ensure_dir(path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)


def run_pca(X, n_components=None, name="", scale=True):
    """
    对特征矩阵 X (N, D) 做 PCA。

    Args:
        X: (N, D) 样本×特征
        n_components: 主成分个数，None 表示取 min(N-1, D)
        name: 用于打印
        scale: 是否先标准化（建议 True，量纲一致）

    Returns:
        pca: 拟合好的 PCA 对象
        X_proj: (N, n_components) 投影后的坐标
        scaler: 若 scale 则返回 StandardScaler，否则 None
    """
    if scale:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    else:
        scaler = None

    n_samples, n_features = X.shape
    if n_components is None:
        n_components = min(n_samples - 1, n_features)
    n_components = min(n_components, n_samples - 1, n_features)

    pca = PCA(n_components=n_components)
    X_proj = pca.fit_transform(X)

    # 汇报用统计
    var_ratio = pca.explained_variance_ratio_
    cumvar = np.cumsum(var_ratio)
    print(f"\n===== {name} PCA 结果 =====")
    print(f"  样本数 N = {X.shape[0]}, 原始维度 D = {X.shape[1]}, 主成分数 = {n_components}")
    print(f"  前 5 个主成分方差解释率: {var_ratio[:5].round(4).tolist()}")
    print(f"  前 5 个主成分累计方差:   {cumvar[:5].round(4).tolist()}")
    print(f"  前 2 维累计方差: {cumvar[1]:.4f} (画 PC1 vs PC2 时能解释的信息比例)")

    return pca, X_proj, scaler


def plot_pca_results(pca, X_proj, labels=None, out_path=None, title_prefix=""):
    """
    画两张图：方差解释率曲线 + PC1 vs PC2 散点图。
    labels: 可选 (N,) 如人物 ID 或视频 ID，用于着色/图例
    """
    if plt is None:
        print("未安装 matplotlib，跳过画图")
        return

    var_ratio = pca.explained_variance_ratio_
    cumvar = np.cumsum(var_ratio)
    n_show = min(20, len(var_ratio))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 左图：方差解释
    axes[0].bar(range(1, n_show + 1), var_ratio[:n_show], alpha=0.7, label="单成分")
    axes[0].plot(range(1, n_show + 1), cumvar[:n_show], "o-", color="C1", label="累计")
    axes[0].set_xlabel("主成分序号")
    axes[0].set_ylabel("方差解释率")
    axes[0].set_title(f"{title_prefix} 方差解释率")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 右图：PC1 vs PC2
    if labels is not None:
        uniq = np.unique(labels)
        for i, u in enumerate(uniq):
            mask = labels == u
            axes[1].scatter(
                X_proj[mask, 0], X_proj[mask, 1],
                alpha=0.6, s=20, label=str(u)
            )
    else:
        axes[1].scatter(X_proj[:, 0], X_proj[:, 1], alpha=0.6, s=20)
    axes[1].set_xlabel(f"PC1 ({var_ratio[0]:.2%})")
    axes[1].set_ylabel(f"PC2 ({var_ratio[1]:.2%})")
    axes[1].set_title(f"{title_prefix} PC1 vs PC2")
    if labels is not None and len(uniq) <= 20:
        axes[1].legend(loc="best", fontsize=8)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    if out_path:
        _ensure_dir(out_path)
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"  已保存: {out_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="对 personality / motion 特征做 PCA，并生成汇报用图表与统计"
    )
    parser.add_argument(
        "personality_npy",
        type=str,
        help="personality 特征 .npy 路径，shape (N, 512)",
    )
    parser.add_argument(
        "motion_npy",
        type=str,
        help="motion 特征 .npy 路径，shape (N, 20) 或 (N, 512)。建议用 20 维便于解释。",
    )
    parser.add_argument(
        "--labels",
        type=str,
        default=None,
        help="可选：标签 .npy 路径，shape (N,)（如人物/视频 ID），用于散点图着色",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="./pca_outputs",
        help="输出目录：图表与 summary 文本",
    )
    parser.add_argument(
        "--n-components",
        type=int,
        default=None,
        help="主成分个数，默认用 min(N-1, D)",
    )
    args = parser.parse_args()

    personality = np.load(args.personality_npy).astype(np.float64)
    motion = np.load(args.motion_npy).astype(np.float64)

    if personality.shape[0] != motion.shape[0]:
        raise ValueError(
            f"personality 与 motion 样本数不一致: {personality.shape[0]} vs {motion.shape[0]}"
        )
    N = personality.shape[0]
    labels = np.load(args.labels) if args.labels else None
    if labels is not None and len(labels) != N:
        raise ValueError(f"labels 长度 {len(labels)} 与样本数 {N} 不一致")

    os.makedirs(args.out_dir, exist_ok=True)

    # Personality PCA
    pca_p, Xp, _ = run_pca(
        personality,
        n_components=args.n_components,
        name="Personality (身份/风格)",
        scale=True,
    )
    plot_pca_results(
        pca_p, Xp,
        labels=labels,
        out_path=os.path.join(args.out_dir, "personality_pca.png"),
        title_prefix="Personality",
    )

    # Motion PCA
    pca_m, Xm, _ = run_pca(
        motion,
        n_components=args.n_components,
        name="Motion (动作)",
        scale=True,
    )
    plot_pca_results(
        pca_m, Xm,
        labels=labels,
        out_path=os.path.join(args.out_dir, "motion_pca.png"),
        title_prefix="Motion",
    )

    # 写一份汇报要点到文本
    summary_path = os.path.join(args.out_dir, "report_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("=== Personality 与 Motion PCA 分析 — 汇报要点 ===\n\n")
        f.write("1) 数据与预处理\n")
        f.write(f"   样本数 N = {N}（每样本一帧 face 图像）。\n")
        f.write(f"   Personality 原始维度 512；Motion 原始维度 {motion.shape[1]}。\n")
        f.write("   做 PCA 前对特征做了标准化（零均值、单位方差）。\n\n")
        f.write("2) Personality PCA\n")
        vp = pca_p.explained_variance_ratio_
        cp = np.cumsum(vp)
        f.write(f"   前 2 维累计方差: {cp[1]:.4f}。\n")
        f.write("   若同一人不同帧的 personality 点聚在一起，说明身份/风格在表示空间里较稳定、可区分。\n\n")
        f.write("3) Motion PCA\n")
        vm = pca_m.explained_variance_ratio_
        cm = np.cumsum(vm)
        f.write(f"   前 2 维累计方差: {cm[1]:.4f}。\n")
        f.write("   若不同动作在 PC1/PC2 上分布不同，说明 20 维 motion 空间能区分不同动作模式。\n\n")
        f.write("4) 结论建议（可口头表述）\n")
        f.write("   - 模型把 face 解耦成 personality（身份）和 motion（动作）两路特征。\n")
        f.write("   - 对两路分别做 PCA 可以看：身份是否聚类、动作是否在低维空间有结构。\n")
        f.write("   - 若 personality 前几维方差集中，说明身份信息较紧凑；motion 同理。\n")
    print(f"\n汇报要点已写入: {summary_path}")


if __name__ == "__main__":
    main()
