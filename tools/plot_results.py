"""plot_results.py — 把扫参/对比的 CSV 画成申报书可直接用的图

用法：
  python plot_results.py            # 同时画两张（文件存在才画）
  产物：
  - fig_tau_curve.png      τ 扫参权衡曲线（双 y 轴：时延 + 上云比例）
  - fig_model_compare.png  多模型对比柱状图（mAP50 / Recall / 推理时延）

依赖：pip install matplotlib pandas（环境里应该都有）
中文显示：Windows 自动用微软雅黑，无需额外配置
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # 无窗口环境也能出图
import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False


def plot_tau(csv_path="results/tau_summary.csv", out="figures/fig_tau_curve.png"):
    if not Path(csv_path).exists():
        print(f"[跳过] {csv_path} 不存在（先跑 route_v2.py）")
        return
    df = pd.read_csv(csv_path)

    tau_rows = df[df["方案"].str.startswith("tau=")].copy()
    tau_rows["tau"] = tau_rows["方案"].str.replace("tau=", "").astype(float)
    tau_rows = tau_rows.sort_values("tau")
    a = df[df["方案"] == "A_全边缘"].iloc[0]
    b = df[df["方案"] == "B_全上云"].iloc[0]

    fig, ax1 = plt.subplots(figsize=(8, 5))

    # 左轴：时延
    ax1.plot(tau_rows["tau"], tau_rows["p50_ms"], "o-", color="#1565C0", label="p50 时延")
    ax1.plot(tau_rows["tau"], tau_rows["p95_ms"], "s-", color="#E53935", label="p95 时延")
    ax1.axhline(a["p50_ms"], ls="--", lw=1, color="#888", alpha=0.6)
    ax1.axhline(b["p50_ms"], ls=":", lw=1, color="#888", alpha=0.6)
    ax1.text(tau_rows["tau"].min(), a["p50_ms"], "  全边缘 p50", va="bottom", fontsize=9, color="#666")
    ax1.text(tau_rows["tau"].min(), b["p50_ms"], "  全上云 p50", va="bottom", fontsize=9, color="#666")
    ax1.set_xlabel("分流阈值 τ")
    ax1.set_ylabel("端到端时延 (ms)")
    ax1.set_ylim(0, max(b["p95_ms"], tau_rows["p95_ms"].max()) * 1.15)

    # 右轴：上云比例
    ax2 = ax1.twinx()
    ax2.plot(tau_rows["tau"], tau_rows["上云比例"], "^--", color="#00897B", label="上云比例")
    ax2.set_ylabel("上云比例")
    ax2.set_ylim(0, 1)

    # 合并图例
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=9)

    ax1.set_title("置信度路由：τ 扫参权衡曲线（NEU-DET 测试集）")
    ax1.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    print(f"[完成] {out}")


def plot_compare(csv_path="results/compare_summary.csv", out="figures/fig_model_compare.png"):
    if not Path(csv_path).exists():
        print(f"[跳过] {csv_path} 不存在（先跑 model_compare_v2.py）")
        return
    df = pd.read_csv(csv_path)
    df = df.sort_values("mAP50", ascending=False)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    items = [
        ("mAP50", "mAP@0.5", "#1565C0", True),
        ("Recall", "Recall（查全率）", "#00897B", True),
        ("infer_ms_per_img", "CPU 推理时延 (ms/张)", "#EF6C00", False),
    ]
    for ax, (col, label, color, higher_better) in zip(axes, items):
        vals = df[col].to_numpy()
        best_pos = int(vals.argmax()) if higher_better else int(vals.argmin())
        bars = ax.bar(df["model"], vals, color=color, alpha=0.85)
        bars[best_pos].set_edgecolor("red")
        bars[best_pos].set_linewidth(2)
        for bar, v in zip(bars, df[col]):
            ax.text(bar.get_x() + bar.get_width() / 2, v, f"{v:g}",
                    ha="center", va="bottom", fontsize=8)
        ax.set_title(label, fontsize=12)
        ax.grid(axis="y", alpha=0.3)
        ax.tick_params(axis="x", rotation=20)

    fig.suptitle("多模型对比（NEU-DET 测试集，红框为最优）", fontsize=13)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    print(f"[完成] {out}")


if __name__ == "__main__":
    only = sys.argv[1] if len(sys.argv) > 1 else "all"
    if only in ("all", "tau"):
        plot_tau()
    if only in ("all", "compare"):
        plot_compare()
