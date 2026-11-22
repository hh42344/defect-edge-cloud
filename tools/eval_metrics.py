# -*- coding: utf-8 -*-
"""tools/eval_metrics.py -- 逐框评估：Recall / FPPI / FROC

作用
    1. 跑一遍边缘模型，把【每一个框】的坐标/类别/置信度存下来（cache_v2.json）
    2. 和 GT 按 IoU 匹配，逐框判定 TP / FP / FN
    3. 扫阈值，算 Recall / FPPI / Precision，画 FROC 曲线

为什么必须逐框
    以前的 cache.json 每张图只存了 max(conf)，没有坐标、没有类别、没有逐框置信度。
    没有逐框数据，就算不出"哪一类漏了几个"、"每张图冤枉几个"，
    也就无法做 ROI 裁剪和路由。

名词
    TP  报对了的框          FP  冤枉的框          FN  漏掉的 GT
    Recall = TP / (TP + FN)         该抓的抓到几成
    FPPI   = FP / 图数              平均每张图冤枉几个（没有负样本也能算）
    FROC   横轴 FPPI、纵轴 Recall 的曲线，工业/医学检测的标准图

用法
    python tools/eval_metrics.py
    python tools/eval_metrics.py --model models/YOLOv9t/yolov9t_best.onnx --imgsz 640
    python tools/eval_metrics.py --split val --iou 0.5 --conf 0.01

产出
    cache_v2.json            逐框推理结果
    froc.csv                 每个阈值下的 Recall / FPPI / Precision
    metrics_by_class.csv     各类在指定工作点的 Recall / FPPI
    fig_froc.png             FROC 曲线
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


def iou_one(a, b) -> float:
    """单个框对单个框的 IoU。a/b 都是 (x1,y1,x2,y2)。"""
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, x2 - x1), max(0.0, y2 - y1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / (area_a + area_b - inter + 1e-9)


def load_gt(txt: Path, w: int, h: int):
    """YOLO 标签 -> [(cls, x1,y1,x2,y2), ...]（像素坐标）"""
    out = []
    if not txt.exists():
        return out
    for line in txt.read_text(encoding="utf-8", errors="ignore").splitlines():
        p = line.split()
        if len(p) < 5:
            continue
        c = int(float(p[0]))
        cx, cy, bw, bh = (float(v) for v in p[1:5])
        out.append((c, (cx - bw / 2) * w, (cy - bh / 2) * h,
                    (cx + bw / 2) * w, (cy + bh / 2) * h))
    return out


def match_image(preds, gts, iou_thr: float):
    """贪心匹配：预测按置信度降序，每个预测去找同类、未匹配、IoU 最大的 GT。
    preds: [(cls, conf, box), ...]  gts: [(cls, box), ...]
    返回 (preds_with_flag, n_gt, n_matched)
    """
    matched = [False] * len(gts)
    flags = []
    for cls, conf, box in sorted(preds, key=lambda x: -x[1]):
        best_i, best_v = -1, iou_thr
        for i, (gc, gb) in enumerate(gts):
            if matched[i] or gc != cls:
                continue
            v = iou_one(box, gb)
            if v >= best_v:
                best_i, best_v = i, v
        if best_i >= 0:
            matched[best_i] = True
            flags.append((cls, conf, True))
        else:
            flags.append((cls, conf, False))
    return flags, len(gts), sum(matched)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/YOLOv9t/yolov9t_best.onnx")
    ap.add_argument("--root", default=".")
    ap.add_argument("--split", default="val")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.01, help="推理时的最低置信度")
    ap.add_argument("--iou", type=float, default=0.5, help="匹配用的 IoU 门槛")
    ap.add_argument("--work-conf", type=float, default=0.30, help="报告用的工作点")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    img_dir = root / "images" / args.split
    lbl_dir = root / "labels" / args.split
    if not img_dir.is_dir():
        print(f"[错误] 找不到 {img_dir}")
        return 1

    from ultralytics import YOLO
    model = YOLO(str(root / args.model))

    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    imgs = sorted([p for p in img_dir.iterdir() if p.suffix.lower() in exts])
    if not imgs:
        print(f"[错误] {img_dir} 下没有图片")
        return 1
    print(f"模型: {args.model} | 图数: {len(imgs)} | imgsz: {args.imgsz} | IoU门槛: {args.iou}")

    # 预热，避免首图冷启动污染（时延统计才需要，这里顺手做）
    for p in imgs[:2]:
        model(str(p), imgsz=args.imgsz, conf=args.conf, verbose=False)

    records = []
    all_flags = []          # [(cls, conf, is_tp)]
    per_img_gt = []         # 每张图的 GT 数
    n_img_no_gt = 0

    for i, p in enumerate(imgs, 1):
        r = model(str(p), imgsz=args.imgsz, conf=args.conf, verbose=False)[0]
        h, w = r.orig_shape[:2]
        if len(r.boxes):
            xyxy = r.boxes.xyxy.cpu().numpy()
            cls = r.boxes.cls.cpu().numpy().astype(int)
            cf = r.boxes.conf.cpu().numpy()
            preds = [(int(c), float(f), tuple(map(float, b))) for c, f, b in zip(cls, cf, xyxy)]
        else:
            preds = []

        gts = load_gt(lbl_dir / (p.stem + ".txt"), w, h)
        if not gts:
            n_img_no_gt += 1
        flags, n_gt, n_match = match_image(preds, [(g[0], g[1:]) for g in gts], args.iou)
        all_flags.extend(flags)
        per_img_gt.append(n_gt)

        records.append({
            "img": p.name, "w": w, "h": h, "n_gt": n_gt,
            "preds": [{"cls": CLASS_NAMES[c] if c < len(CLASS_NAMES) else str(c),
                       "conf": round(f, 4),
                       "xyxy": [round(v, 1) for v in b]} for c, f, b in preds],
            "n_tp": int(sum(1 for _, _, t in flags if t)),
        })
        if i % 50 == 0 or i == len(imgs):
            print(f"  [{i}/{len(imgs)}] 已处理")

    n_img = len(records)
    n_gt_total = sum(per_img_gt)
    (root / "results" / "cache").mkdir(parents=True, exist_ok=True)
    (root / "figures").mkdir(parents=True, exist_ok=True)
    (root / "results" / "cache" / "cache_v2.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n已写入 results/cache_v2.json（逐框，{n_img} 张图 / {n_gt_total} 个 GT）")

    # ---------- 扫阈值 ----------
    confs = np.array([f[1] for f in all_flags]) if all_flags else np.array([])
    tps = np.array([f[2] for f in all_flags], dtype=bool) if all_flags else np.array([], bool)
    clss = np.array([f[0] for f in all_flags]) if all_flags else np.array([], int)

    thresholds = np.round(np.arange(0.01, 0.96, 0.01), 2)
    rows = []
    for t in thresholds:
        keep = confs >= t
        tp = int((tps & keep).sum())
        fp = int((~tps & keep).sum())
        fn = n_gt_total - tp
        rows.append({
            "conf": float(t),
            "TP": tp, "FP": fp, "FN": fn,
            "Recall": round(tp / n_gt_total, 4) if n_gt_total else 0.0,
            "Precision": round(tp / (tp + fp), 4) if (tp + fp) else 0.0,
            "FPPI": round(fp / n_img, 4),
        })

    with open(root / "results" / "froc.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # ---------- 分类别指标（固定工作点） ----------
    cls_rows = []
    # 统计每个类别的 GT 数
    gt_per_cls = {i: 0 for i in range(len(CLASS_NAMES))}
    for p in imgs:
        r0 = next(x for x in records if x["img"] == p.name)
        for c, *_ in load_gt(lbl_dir / (p.stem + ".txt"), r0["w"], r0["h"]):
            gt_per_cls[c] = gt_per_cls.get(c, 0) + 1
    tp_per_cls = {i: 0 for i in range(len(CLASS_NAMES))}
    fp_per_cls = {i: 0 for i in range(len(CLASS_NAMES))}
    for c, f, is_tp in zip(clss, confs, tps):
        if f < args.work_conf:
            continue
        if is_tp:
            tp_per_cls[int(c)] = tp_per_cls.get(int(c), 0) + 1
        else:
            fp_per_cls[int(c)] = fp_per_cls.get(int(c), 0) + 1

    print(f"\n工作点 conf >= {args.work_conf} 的分类别指标")
    print(f"{'类别':<16}{'GT':>6}{'TP':>6}{'FP':>6}{'漏':>6}{'Recall':>9}{'FPPI':>9}")
    print("-" * 60)
    for ci, name in enumerate(CLASS_NAMES):
        g = gt_per_cls.get(ci, 0)
        tp = tp_per_cls.get(ci, 0)
        fp = fp_per_cls.get(ci, 0)
        fn = g - tp
        rec = tp / g if g else 0.0
        print(f"{name:<16}{g:>6}{tp:>6}{fp:>6}{fn:>6}{rec:>9.3f}{fp / n_img:>9.3f}")
        cls_rows.append({"class": name, "GT": g, "TP": tp, "FP": fp, "FN": fn,
                         "Recall": round(rec, 4), "FPPI": round(fp / n_img, 4)})

    with open(root / "results" / "metrics_by_class.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(cls_rows[0].keys()))
        w.writeheader()
        w.writerows(cls_rows)

    # ---------- 总体 + FROC 图 ----------
    tp = sum(r["TP"] for r in rows if abs(r["conf"] - args.work_conf) < 1e-9)
    print(f"\n总体：{n_img} 张图 / {n_gt_total} 个 GT / {n_img_no_gt} 张无标注")
    for t in (0.05, 0.15, 0.30, 0.50):
        r = min(rows, key=lambda x: abs(x["conf"] - t))
        print(f"  conf>={r['conf']:.2f}   Recall={r['Recall']:.3f}   "
              f"FPPI={r['FPPI']:.3f}   Precision={r['Precision']:.3f}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        xs = [r["FPPI"] for r in rows]
        ys = [r["Recall"] for r in rows]
        fig, ax = plt.subplots(figsize=(6.4, 4.8), dpi=150)
        ax.plot(xs, ys, "-", color="#c0392b", lw=2, label="YOLOv9t (edge)")
        ax.axvline(1.0, ls="--", color="gray", lw=1)
        ax.text(1.02, 0.05, "FPPI = 1.0", color="gray", fontsize=8)
        ax.set_xlabel("FPPI  (false positives per image)")
        ax.set_ylabel("Recall")
        ax.set_title("FROC  —  edge model on NEU-DET val")
        ax.set_xlim(0, max(5.0, max(xs) if xs else 5.0))
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.3)
        ax.legend(loc="lower right")
        fig.tight_layout()
        fig.savefig(root / "figures" / "fig_froc.png")
        print("\n已写入 figures/fig_froc.png、results/froc.csv、results/metrics_by_class.csv")
    except Exception as e:
        print(f"[提示] 画图失败（{e}），但 CSV 已写好")

    return 0


if __name__ == "__main__":
    sys.exit(main())
