# -*- coding: utf-8 -*-
"""tools/plot_baseline.py -- 路由参考基线（预算-召回曲线）

做什么
    从已有的 cache_v2.json（逐框推理结果）出发，不重跑模型，
    直接模拟"每图送 k 个候选框去云端复检"的四种挑选策略，
    画出 Recall ~ k 的基线曲线。

四条线
    random      随机挑 k 个             —— 下界，证明"不是随便挑都行"
    conf_asc    挑置信度最低的 k 个      —— 传统阈值法思路
    conf_desc   挑置信度最高的 k 个      —— 【主要对手】单特征最优
    oracle      只从真缺陷里挑 k 个      —— 理论上界，衡量还剩多少空间

系统模型（与方案一致）
    conf >= TAU_HIGH           → 边缘直接输出（不送云）
    TAU_LOW <= conf < TAU_HIGH → 候选池，路由挑 k 个送云
    被挑中的 → 云端判决：真缺陷则保留，背景则驳回
    未被挑中的候选框        → 丢弃（不报，避免误报）

云端代理由 GT 充任（oracle 假设）
    is_tp 为真 → 云端"确认"；否则"驳回"。
    因此这是【理论上界】。等分类器训好，把 confirmed 换成真实判决即为实测曲线。

用法
    python tools\\plot_baseline.py
    python tools\\plot_baseline.py --tau-high 0.5 --tau-low 0.05 --kmax 5

产出
    results/baseline.csv        每档 k × 每个策略的 Recall / FPPI / 送检数
    figures/fig_baseline.png    四条基线曲线
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

CLASS_NAMES = ["crazing", "inclusion", "patches",
               "pitted_surface", "rolled-in_scale", "scratches"]
STRATEGIES = ["random", "conf_asc", "conf_desc", "oracle"]


def iou_one(a, b) -> float:
    """单框 IoU，a/b 均为 (x1,y1,x2,y2)"""
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, x2 - x1), max(0.0, y2 - y1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    aa = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    ab = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / (aa + ab - inter + 1e-9)


def load_gt(txt: Path, w: int, h: int):
    """YOLO 标签 -> [(类别名, (x1,y1,x2,y2))]，像素坐标"""
    out = []
    if not txt.exists():
        return out
    for line in txt.read_text(encoding="utf-8", errors="ignore").splitlines():
        p = line.split()
        if len(p) < 5:
            continue
        c = int(float(p[0]))
        cx, cy, bw, bh = (float(v) for v in p[1:5])
        name = CLASS_NAMES[c] if 0 <= c < len(CLASS_NAMES) else str(c)
        out.append((name, ((cx - bw / 2) * w, (cy - bh / 2) * h,
                           (cx + bw / 2) * w, (cy + bh / 2) * h)))
    return out


def match_flags(preds, gts, thr: float):
    """贪心匹配：预测按置信度降序，找同类未匹配 IoU 最大的 GT。
    preds: [{"cls": name, "conf": p, "xyxy": [...]}]  gts: [(name, box)]
    返回与 preds 等长的 bool 列表（True = TP）
    """
    order = sorted(range(len(preds)), key=lambda i: -float(preds[i]["conf"]))
    used = [False] * len(gts)
    flags = [False] * len(preds)
    for i in order:
        pc = preds[i]["cls"]
        box = preds[i]["xyxy"]
        best_j, best_v = -1, thr
        for j, (gc, gb) in enumerate(gts):
            if used[j] or gc != pc:
                continue
            v = iou_one(box, gb)
            if v >= best_v:
                best_j, best_v = j, v
        if best_j >= 0:
            used[best_j] = True
            flags[i] = True
    return flags


def pick_indices(cand, k: int, strategy: str, rng=None):
    """从候选池挑 k 个的索引"""
    n = len(cand)
    if k <= 0 or n == 0:
        return []
    k = min(k, n)
    if strategy == "random":
        return list(rng.choice(n, size=k, replace=False))
    if strategy == "conf_asc":
        return sorted(range(n), key=lambda i: cand[i]["conf"])[:k]
    if strategy == "conf_desc":
        return sorted(range(n), key=lambda i: -cand[i]["conf"])[:k]
    if strategy == "oracle":
        # 只从 TP 里挑，TP 不够才用 FP 凑（不改变召回）
        tps = [i for i in range(n) if cand[i]["is_tp"]]
        fps = [i for i in range(n) if not cand[i]["is_tp"]]
        return (tps + fps)[:k]
    raise ValueError(strategy)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--cache", default="results/cache/cache_v2.json")
    ap.add_argument("--split", default="val", help="标签目录，用于取 GT")
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--tau-high", type=float, default=0.50, help="直接输出的置信度门槛")
    ap.add_argument("--tau-low", type=float, default=0.05, help="候选池的置信度下限")
    ap.add_argument("--kmax", type=int, default=5, help="每图最多送几个")
    ap.add_argument("--runs", type=int, default=20, help="random 策略重复次数")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    root = Path(args.root).resolve()
    cache_path = root / args.cache
    if not cache_path.exists():
        print(f"[错误] 找不到 {cache_path}\n先跑：python tools\\eval_metrics.py")
        return 1

    records = json.loads(cache_path.read_text(encoding="utf-8"))
    lbl_dir = root / "labels" / args.split

    # ---------- 逐图切分：高置信 / 候选池 ----------
    per_img = []       # [(high_tp, high_fp, [(conf, is_tp), ...])]
    n_gt_total = 0
    n_pred_total = 0

    for rec in records:
        gts = load_gt(lbl_dir / (Path(rec["img"]).stem + ".txt"), rec["w"], rec["h"])
        n_gt_total += len(gts)
        preds = rec.get("preds", [])
        n_pred_total += len(preds)
        flags = match_flags(preds, gts, args.iou)

        high, cand = [], []
        for i, b in enumerate(preds):
            c, tp = float(b["conf"]), bool(flags[i])
            if c >= args.tau_high:
                high.append((c, tp))
            elif c >= args.tau_low:
                cand.append({"conf": c, "is_tp": tp})
        high_tp = sum(1 for _, t in high if t)
        high_fp = sum(1 for _, t in high if not t)
        per_img.append((high_tp, high_fp, cand))

    n_img = len(per_img)
    base_tp = sum(h[0] for h in per_img)
    base_fp = sum(h[1] for h in per_img)
    pool_size = sum(len(p[2]) for p in per_img)
    pool_tp = sum(1 for p in per_img for b in p[2] if b["is_tp"])

    print(f"图数 {n_img} | GT {n_gt_total} | 预测框 {n_pred_total}")
    print(f"高置信(>={args.tau_high})：TP {base_tp}  FP {base_fp}  "
          f"→ 纯边缘 Recall {base_tp / n_gt_total:.4f}")
    print(f"候选池({args.tau_low}~{args.tau_high})：{pool_size} 个框，其中 TP {pool_tp} 个 "
          f"（密度 {pool_tp / max(1, pool_size):.1%}），每图平均 {pool_size / n_img:.1f} 个")
    print(f"系统召回上限（全送且云端完美）：{(base_tp + pool_tp) / n_gt_total:.4f}\n")

    # ---------- 扫 k ----------
    rng = np.random.default_rng(args.seed)
    rows = []
    for k in range(0, args.kmax + 1):
        for strat in STRATEGIES:
            if strat == "random":
                acc = np.zeros(args.runs)
                for r in range(args.runs):
                    tp = base_tp + sum(
                        sum(1 for i in pick_indices(c, k, "random", rng) if c[i]["is_tp"])
                        for _, _, c in per_img)
                    acc[r] = tp
                tp = float(acc.mean())
                sd = float(acc.std())
            else:
                tp = base_tp + sum(
                    sum(1 for i in pick_indices(c, k, strat) if c[i]["is_tp"])
                    for _, _, c in per_img)
                sd = 0.0
            sent = sum(min(k, len(c)) for _, _, c in per_img)
            rows.append({
                "k": k,
                "strategy": strat,
                "recall": round(tp / n_gt_total, 4),
                "recall_std": round(sd / n_gt_total, 4),
                "tp": int(round(tp)),
                "fp": base_fp,          # oracle 下送检不产生 FP
                "fppi": round(base_fp / n_img, 4),
                "n_sent": sent,
            })

    # ---------- 落盘 ----------
    (root / "results").mkdir(parents=True, exist_ok=True)
    (root / "figures").mkdir(parents=True, exist_ok=True)
    with open(root / "results" / "baseline.csv", "w", newline="",
              encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(f"{'k':>2} {'random':>8} {'conf_asc':>9} {'conf_desc':>10} {'oracle':>8}")
    print("-" * 42)
    for k in range(0, args.kmax + 1):
        d = {r["strategy"]: r["recall"] for r in rows if r["k"] == k}
        print(f"{k:>2} {d['random']:>8.4f} {d['conf_asc']:>9.4f} "
              f"{d['conf_desc']:>10.4f} {d['oracle']:>8.4f}")

    # ---------- 画图 ----------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        style = {
            "random":    ("#7f8c8d", "o-",  "random"),
            "conf_asc":  ("#2980b9", "s--", "conf ascending (uncertain first)"),
            "conf_desc": ("#e67e22", "^-",  "conf descending (likely first)"),
            "oracle":    ("#27ae60", ":",   "oracle (upper bound)"),
        }
        fig, ax = plt.subplots(figsize=(6.6, 4.6), dpi=150)
        for strat in STRATEGIES:
            xs = [r["k"] for r in rows if r["strategy"] == strat]
            ys = [r["recall"] for r in rows if r["strategy"] == strat]
            col, mk, lab = style[strat]
            ax.plot(xs, ys, mk, color=col, lw=1.8, ms=5, label=lab)
        ax.axhline(base_tp / n_gt_total, ls="-.", lw=1.2, color="#c0392b",
                   label=f"edge only (k=0)")
        ax.set_xlabel("k  (candidates sent to cloud per image)")
        ax.set_ylabel("Recall")
        ax.set_title("Routing baseline  —  budget vs recall (oracle cloud)")
        ax.grid(alpha=0.3)
        ax.legend(loc="lower right", fontsize=8)
        ax.set_xticks(range(0, args.kmax + 1))
        fig.tight_layout()
        fig.savefig(root / "figures" / "fig_baseline.png")
        print("\n已写入 results/baseline.csv 与 figures/fig_baseline.png")
    except Exception as e:
        print(f"[提示] 画图失败（{e}），但 CSV 已写好")

    print("\n注意：本曲线假设云端完美（oracle）。")
    print("分类器训好后，把 confirmed 换成真实判决即为实测曲线。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
