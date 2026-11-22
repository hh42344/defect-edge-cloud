# -*- coding: utf-8 -*-
"""tools/box_stats.py -- GT 框尺寸分布统计

目的
    回答一个决定技术路线的问题：每类缺陷的 GT 框到底有多大？
    是「小目标」（该上 P2 头 / 切片推理 / 高分辨率），
    还是「整片纹理」（该上分类 / 异常检测，改结构纯属浪费）？

原理
    YOLO 标签一行是：cls cx cy w h，四者都归一化到 0~1。
    所以  面积占比 = w * h          （占全图面积的比例，0~1）
          等效边长(px) = sqrt(w*h) * 输入尺寸
    「等效边长」是把这个框当成正方形时的边长，用来跟 32px 这种
    小目标标准直接对比，比裸的 w/h 好读。

用法
    python tools/box_stats.py
    python tools/box_stats.py --root . --out box_stats.csv

产出
    终端表格 + box_stats.csv（UTF-8-BOM，Excel 双击可开）
    每行一类，含：图数/框数/框每图/面积占比中位数与 p10-p90/
    各输入尺寸下的等效边长/五个面积档各自的框数占比(share_area_*)。
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
import time
import unicodedata
from collections import defaultdict
from pathlib import Path

# 与 neu_det.yaml 保持一致；顺序错了统计就错了
DEFAULT_NAMES = ["crazing", "inclusion", "patches",
                 "pitted_surface", "rolled-in_scale", "scratches"]

AREA_BUCKETS = [(0.00, 0.01, "<1%"), (0.01, 0.05, "1-5%"),
                (0.05, 0.20, "5-20%"), (0.20, 0.50, "20-50%"),
                (0.50, 1.01, ">50%")]


def dwidth(s: str) -> int:
    """中文按 2 个字符宽计算，否则终端里表格会歪。"""
    return sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in str(s))


def pad(s, n: int, align: str = "left") -> str:
    s = str(s)
    gap = max(0, n - dwidth(s))
    return (" " * gap + s) if align == "right" else (s + " " * gap)


def pct(x: float, nd: int = 2) -> str:
    return f"{x * 100:.{nd}f}%"


def quantile(xs, q: float):
    """线性插值分位数，不依赖 numpy，避免环境没装就崩。"""
    if not xs:
        return float("nan")
    s = sorted(xs)
    if len(s) == 1:
        return s[0]
    pos = (len(s) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)


def parse_names(root: Path):
    """从 neu_det.yaml 读类名；读不到就用默认顺序。"""
    yml = root / "neu_det.yaml"
    if not yml.exists():
        return DEFAULT_NAMES
    names = {}
    in_names = False
    for line in yml.read_text(encoding="utf-8", errors="ignore").splitlines():
        s = line.strip()
        if s.startswith("names:"):
            in_names = True
            continue
        if in_names:
            if not s or s.startswith("#"):
                continue
            if ":" not in s:
                break
            k, v = s.split(":", 1)
            k, v = k.strip(), v.strip().strip("'\"")
            if k.isdigit():
                names[int(k)] = v
            else:
                break
    return [names[i] for i in sorted(names)] if names else DEFAULT_NAMES


def probe_img_size(root: Path, default: int = 200) -> int:
    """取一张图的边长；没有 PIL 就按 NEU-DET 的 200x200 处理。"""
    try:
        from PIL import Image
    except Exception:
        print(f"[提示] 没装 PIL，按 NEU-DET 默认 {default}x{default} 处理")
        return default
    for sub in ("train", "val", "test"):
        d = root / "images" / sub
        if d.is_dir():
            for f in d.iterdir():
                if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
                    try:
                        with Image.open(f) as im:
                            if im.width != im.height:
                                print(f"[警告] 图像非正方形: {f.name} {im.size}")
                            return im.width
                    except Exception:
                        continue
    return default


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".", help="项目根目录（含 labels/）")
    ap.add_argument("--out", default="results/box_stats.csv")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    lbl_root = root / "labels"
    if not lbl_root.is_dir():
        print(f"[错误] 找不到 {lbl_root}")
        return 1

    names = parse_names(root)
    img_side = probe_img_size(root)

    areas = defaultdict(list)       # cls_name -> [area_ratio, ...]
    boxes_per_img = defaultdict(list)
    imgs_per_cls = defaultdict(set)
    empty_labels = []
    bad_lines = 0
    splits = sorted([d.name for d in lbl_root.iterdir() if d.is_dir()])

    for split in splits:
        for f in sorted((lbl_root / split).glob("*.txt")):
            rows = []
            for ln, line in enumerate(f.read_text(encoding="utf-8", errors="ignore").splitlines(), 1):
                p = line.split()
                if not p:
                    continue
                if len(p) < 5:
                    bad_lines += 1
                    continue
                try:
                    cid = int(float(p[0]))
                    w, h = float(p[3]), float(p[4])
                except ValueError:
                    bad_lines += 1
                    continue
                if cid < 0 or cid >= len(names):
                    bad_lines += 1
                    continue
                rows.append((names[cid], max(0.0, min(1.0, w * h))))
            if not rows:
                empty_labels.append(f"{split}/{f.stem}")
                continue
            for cname, a in rows:
                areas[cname].append(a)
                imgs_per_cls[cname].add(f"{split}/{f.stem}")
            for cname in {r[0] for r in rows}:
                boxes_per_img[cname].append(sum(1 for r in rows if r[0] == cname))

    if not areas:
        print("[错误] 一条有效标注都没解析到，检查 labels 目录结构")
        return 1

    # ---------------- 表 1：每类整体尺寸 ----------------
    head = ["类别", "图数", "框数", "框/图", "面积占比", "p10", "p90",
            f"边长@{img_side}", "边长@640", "边长@960"]
    widths = [18, 6, 6, 6, 9, 7, 7, 11, 10, 10]
    print(f"\n数据集: {len(set(x for s in imgs_per_cls.values() for x in s))} 图 / "
          f"标注目录 {splits} / 图像边长 {img_side}px")
    print("=" * (sum(widths) + len(widths)))
    print(" ".join(pad(h, w) for h, w in zip(head, widths)))
    print("-" * (sum(widths) + len(widths)))

    rows_out = []
    for cname in names:
        a = areas.get(cname, [])
        if not a:
            continue
        med = quantile(a, 0.5)
        side = lambda r: int(round(math.sqrt(med) * r))
        bpi = boxes_per_img[cname]
        line = [cname, len(imgs_per_cls[cname]), len(a),
                f"{sum(bpi) / len(bpi):.2f}", pct(med, 1),
                pct(quantile(a, 0.1), 1), pct(quantile(a, 0.9), 1),
                f"{side(img_side)}px", f"{side(640)}px", f"{side(960)}px"]
        print(" ".join(pad(v, w, "right" if i else "left")
                       for i, (v, w) in enumerate(zip(line, widths))))
        # 面积分档：该档框数 / 本类总框数，必须进 CSV，否则终端一关数据就没了
        shares = {}
        for lo, hi, tag in AREA_BUCKETS:
            key = tag.replace("%", "").replace("<", "lt").replace(">", "gt").replace("-", "_")
            shares[f"share_area_{key}"] = round(sum(1 for x in a if lo <= x < hi) / len(a), 4)

        rows_out.append({
            "class": cname, "images": len(imgs_per_cls[cname]), "boxes": len(a),
            "boxes_per_img": round(sum(bpi) / len(bpi), 3),
            "area_median": round(med, 4),
            "area_p10": round(quantile(a, 0.1), 4),
            "area_p90": round(quantile(a, 0.9), 4),
            "side_at_%d" % img_side: side(img_side),
            "side_at_640": side(640), "side_at_960": side(960),
            **shares,
        })

    # ---------------- 表 2：面积分档 ----------------
    print("\n面积占比分档（该档框数占本类框数的比例）")
    bh = ["类别"] + [b[2] for b in AREA_BUCKETS]
    bw = [18] + [10] * len(AREA_BUCKETS)
    print(" ".join(pad(h, w) for h, w in zip(bh, bw)))
    print("-" * (sum(bw) + len(bw)))
    for cname in names:
        a = areas.get(cname, [])
        if not a:
            continue
        cells = []
        for lo, hi, _ in AREA_BUCKETS:
            cells.append(pct(sum(1 for x in a if lo <= x < hi) / len(a), 1))
        print(" ".join(pad(v, w, "right" if i else "left")
                       for i, (v, w) in enumerate(zip([cname] + cells, bw))))

    # ---------------- 表 3：结论 ----------------
    print("\n" + "=" * 60)
    print("结论（按面积占比中位数判定，仅对该数据集成立）")
    for cname in names:
        a = areas.get(cname, [])
        if not a:
            continue
        med = quantile(a, 0.5)
        if med < 0.02:
            v = "小目标 (<2%) -> P2头/切片/高分辨率都值得试"
        elif med < 0.10:
            v = "中等目标 (2-10%) -> 优先调输入尺寸，P2头收益有限"
        elif med < 0.30:
            v = "大型目标 (10-30%) -> P2头无效；瓶颈是特征/语义，不是分辨率"
        else:
            v = "整片纹理 (>30%) -> 这是分类/异常检测问题，不是检测问题"
        print(f"  {pad(cname, 18)}{pct(med, 1):>7}  {v}")

    if empty_labels:
        print(f"\n[警告] {len(empty_labels)} 个标注文件为空（会被当背景图）：")
        print("  " + ", ".join(empty_labels[:10]) + (" ..." if len(empty_labels) > 10 else ""))
    if bad_lines:
        print(f"[警告] {bad_lines} 行标注格式异常，已跳过")

    # ---------------- 落盘 ----------------
    out = root / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        fh = open(out, "w", newline="", encoding="utf-8-sig")
    except PermissionError:
        # 常见于用 Excel 打开了这个 csv，文件被锁；换个名字写，别让统计白跑
        fallback = out.with_name(f"{out.stem}_{time.strftime('%Y%m%d_%H%M%S')}{out.suffix}")
        print(f"[提示] {out.name} 被占用（多半是 Excel 开着），改写到 {fallback.name}")
        out, fh = fallback, open(fallback, "w", newline="", encoding="utf-8-sig")
    with fh as f:
        wri = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
        wri.writeheader()
        wri.writerows(rows_out)
    print(f"\n已写入 {out}")

    print("\n下一步：把 crazing 那一行的『面积占比中位数』发我，")
    print("它决定 P2 头、切片推理、高分辨率复检这三件事要不要继续做。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
