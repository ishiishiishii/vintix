#!/usr/bin/env python3
"""
補助解析用: 専門家データと Vintix 評価ログを比較し、分布のズレを可視化する。

このスクリプトは検証目的のための確認コードであり、本研究の中核部分ではない。
"""

import argparse
import math
from pathlib import Path

import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")


def load_observations_from_expert(path: Path,
                                  key: str = "proprio_observation",
                                  max_samples: int = 200_000,
                                  seed: int = 0) -> np.ndarray:
    """専門家データ (HDF5) から観測を読み込む。"""
    rng = np.random.default_rng(seed)
    samples = []
    total = 0
    with h5py.File(path, "r") as f:
        group_names = sorted(f.keys(), key=lambda s: int(s.split("-")[0]))
        for gname in group_names:
            data = f[gname][key][:]
            if data.size == 0:
                continue
            remaining = max_samples - total
            if remaining <= 0:
                break
            if data.shape[0] <= remaining:
                samples.append(data)
                total += data.shape[0]
            else:
                idx = rng.choice(data.shape[0], size=remaining, replace=False)
                samples.append(data[idx])
                total += remaining
    if not samples:
        raise ValueError("No data loaded from expert dataset.")
    return np.concatenate(samples, axis=0)


def load_eval_observations(path: Path,
                           key: str = "proprio_observation") -> np.ndarray:
    """評価ログ (confirm_eval_history_recorder で生成) から観測を読み込む。"""
    with h5py.File(path, "r") as f:
        group_names = sorted(f.keys(), key=lambda s: int(s.split("-")[0]))
        samples = [f[gname][key][:] for gname in group_names]
    if not samples:
        raise ValueError("No data loaded from eval dataset.")
    return np.concatenate(samples, axis=0)


def compute_pca(expert_data: np.ndarray, n_components: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """PCA (共分散行列から固有値分解) を計算し、主成分を返す。"""
    centered = expert_data - expert_data.mean(axis=0, keepdims=True)
    cov = np.cov(centered, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    idx = np.argsort(eigvals)[::-1]
    components = eigvecs[:, idx[:n_components]]
    explained = eigvals[idx[:n_components]] / eigvals.sum()
    return components, explained


def project(data: np.ndarray, mean: np.ndarray, components: np.ndarray) -> np.ndarray:
    """PCA 射影"""
    return (data - mean) @ components


def compute_mahalanobis(data: np.ndarray,
                        mean: np.ndarray,
                        cov: np.ndarray,
                        eps: float = 1e-6) -> np.ndarray:
    """Mahalanobis 距離"""
    cov_reg = cov + np.eye(cov.shape[0]) * eps
    inv_cov = np.linalg.inv(cov_reg)
    diff = data - mean
    return np.sqrt(np.sum(diff @ inv_cov * diff, axis=1))


def plot_results(output_prefix: Path,
                 expert_proj: np.ndarray,
                 eval_proj: np.ndarray,
                 mahal_eval: np.ndarray,
                 rewards_eval: np.ndarray,
                 explained: np.ndarray) -> None:
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Expert vs Eval Distribution Check (auxiliary)", fontsize=16, fontweight="bold")

    # PCA scatter
    ax0 = axes[0, 0]
    ax0.scatter(expert_proj[:, 0], expert_proj[:, 1],
                s=5, alpha=0.2, label="Expert (sampled)", color="#4C72B0")
    ax0.scatter(eval_proj[:, 0], eval_proj[:, 1],
                s=12, alpha=0.6, label="Eval run", color="#DD8452")
    ax0.set_title("PCA Projection (PC1 vs PC2)", fontweight="bold")
    ax0.set_xlabel(f"PC1 ({explained[0]*100:.1f}% var)")
    ax0.set_ylabel(f"PC2 ({explained[1]*100:.1f}% var)")
    ax0.grid(True, alpha=0.3)
    ax0.legend()

    # Mahalanobis distance timeline
    ax1 = axes[0, 1]
    ax1.plot(np.arange(len(mahal_eval)), mahal_eval, color="#DD8452")
    ax1.set_title("Mahalanobis Distance over Eval Steps", fontweight="bold")
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Distance (expert mean/cov)")
    ax1.grid(True, alpha=0.3)

    # Distance histogram
    ax2 = axes[1, 0]
    ax2.hist(mahal_eval, bins=50, color="#DD8452", alpha=0.8)
    ax2.set_title("Mahalanobis Distance Histogram (Eval)", fontweight="bold")
    ax2.set_xlabel("Distance")
    ax2.set_ylabel("Count")

    # Reward vs distance
    ax3 = axes[1, 1]
    ax3.plot(np.arange(len(rewards_eval)), rewards_eval, label="Reward", color="#55A868")
    ax3_twin = ax3.twinx()
    ax3_twin.plot(np.arange(len(mahal_eval)), mahal_eval,
                  label="Mahalanobis", color="#C44E52", alpha=0.5)
    ax3.set_title("Reward vs Mahalanobis Distance (Eval)", fontweight="bold")
    ax3.set_xlabel("Step")
    ax3.set_ylabel("Reward", color="#55A868")
    ax3_twin.set_ylabel("Mahalanobis", color="#C44E52")
    ax3.grid(True, alpha=0.3)
    h1, l1 = ax3.get_legend_handles_labels()
    h2, l2 = ax3_twin.get_legend_handles_labels()
    ax3.legend(h1 + h2, l1 + l2, loc="upper right")

    plt.tight_layout()
    fig.savefig(output_prefix.with_suffix(".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 簡易統計の保存
    summary_text = (
        f"Eval steps: {len(mahal_eval)}\n"
        f"Mahalanobis: mean={mahal_eval.mean():.3f}, "
        f"std={mahal_eval.std():.3f}, "
        f"max={mahal_eval.max():.3f}\n"
        f"Rewards: mean={rewards_eval.mean():.3f}, "
        f"std={rewards_eval.std():.3f}, "
        f"min={rewards_eval.min():.3f}, "
        f"max={rewards_eval.max():.3f}\n"
    )
    summary_path = output_prefix.with_suffix(".txt")
    with open(summary_path, "w") as f:
        f.write("Auxiliary Expert vs Eval Analysis Summary\n")
        f.write(summary_text)


def main():
    parser = argparse.ArgumentParser(description="専門家データと評価履歴の比較 (確認用)")
    parser.add_argument("--expert_path", type=str, required=True, help="専門家データ HDF5")
    parser.add_argument("--eval_path", type=str, required=True, help="評価ログ HDF5")
    parser.add_argument("--output_prefix", type=str, required=True, help="出力ファイルの接頭辞 (PNG/TXT)")
    parser.add_argument("--max_expert_samples", type=int, default=200_000, help="専門家データの最大サンプル数")
    parser.add_argument("--seed", type=int, default=0, help="サンプリング用乱数シード")
    args = parser.parse_args()

    expert_path = Path(args.expert_path)
    eval_path = Path(args.eval_path)
    output_prefix = Path(args.output_prefix)

    print("=" * 80)
    print("Auxiliary Expert vs Eval Analysis")
    print("=" * 80)
    print(f"Expert dataset : {expert_path}")
    print(f"Eval dataset   : {eval_path}")
    print(f"Output prefix  : {output_prefix}")
    print("=" * 80)

    expert_obs = load_observations_from_expert(
        expert_path, max_samples=args.max_expert_samples, seed=args.seed)
    eval_obs = load_eval_observations(eval_path)

    expert_mean = expert_obs.mean(axis=0)
    expert_cov = np.cov((expert_obs - expert_mean), rowvar=False)

    components, explained = compute_pca(expert_obs)
    expert_proj = project(expert_obs, expert_mean, components)
    eval_proj = project(eval_obs, expert_mean, components)

    mahal_eval = compute_mahalanobis(eval_obs, expert_mean, expert_cov)

    # 評価データの報酬を読み込み
    with h5py.File(eval_path, "r") as f:
        rewards_eval = np.concatenate([f[g]["reward"][:] for g in f.keys()], axis=0)

    plot_results(output_prefix, expert_proj, eval_proj, mahal_eval, rewards_eval, explained)
    print(f"✓ Saved analysis to {output_prefix.with_suffix('.png')}")
    print(f"✓ Summary written to {output_prefix.with_suffix('.txt')}")


if __name__ == "__main__":
    main()

