"""route_v2.py — 置信度路由扫参（修正版）

修正点（相对 DeepSeek 版 route.py）：
1. 缓存式扫参：推理只跑两遍（边缘全量 + 云端全量），τ 离线计算，效率提升 ~7x
2. 删除无效的 miss_count 指标（conf<0.3 在 tau>=0.3 时逻辑上不可能触发）
3. 增加云端复检改变判断的计数（flip_count）——这是云端复检有效性的直接信号
4. 增加热身（warmup），消除首图冷启动对时延统计的污染
5. 结果落盘 CSV + 缓存 JSON（供后续精度评估复用）

用法：python route_v2.py
前提：models/YOLOv10n/yolov10n_best.onnx 与云端模型路径存在
"""
import csv, json, time
from pathlib import Path
import numpy as np
from ultralytics import YOLO

# ---------------- 配置 ----------------
EDGE_MODEL = "models/YOLOv10n/yolov10n_best.onnx"   # 边缘端：ONNX + CPU + imgsz 640
CLOUD_MODEL = "models/YOLOv8m/yolov8m_best.pt"    # 云端：FP32 + imgsz 960（换 RT-DETR 时改这里）
TAU_LIST = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
CLOUD_RTT_MS = 80                                  # 模拟边缘→云网络往返
IMAGES_DIR = Path("images/val")                   # 必须是测试集（训练时未见过的图）
WARMUP_N = 3

# ---------------- 加载与热身 ----------------
edge = YOLO(EDGE_MODEL)
cloud = YOLO(CLOUD_MODEL)
imgs = sorted(list(IMAGES_DIR.glob("*.jpg")) + list(IMAGES_DIR.glob("*.png")))
assert imgs, f"{IMAGES_DIR} 下没有图片"

print(f"边缘模型: {EDGE_MODEL} | 云端模型: {CLOUD_MODEL} | 图片数: {len(imgs)}")
for w in imgs[:WARMUP_N]:  # 热身：消除首次调用的初始化开销
    edge(w, device="cpu", imgsz=640, verbose=False)
    cloud(w, device="cpu", imgsz=960, verbose=False)

# ---------------- 阶段一：全量推理各一遍，缓存逐图数据 ----------------
records = []
for i, img in enumerate(imgs):
    t0 = time.perf_counter()
    r = edge(img, device="cpu", conf=0.01, imgsz=640, verbose=False)[0]
    t_edge_ms = (time.perf_counter() - t0) * 1000
    conf = float(r.boxes.conf.max()) if len(r.boxes) else 0.0
    n_edge_boxes = len(r.boxes)

    t1 = time.perf_counter()
    rc = cloud(img, device="cpu", conf=0.01, imgsz=960, verbose=False)[0]
    t_cloud_ms = (time.perf_counter() - t1) * 1000
    conf_c = float(rc.boxes.conf.max()) if len(rc.boxes) else 0.0

    records.append({"img": img.name, "conf_edge": conf, "conf_cloud": conf_c,
                    "edge_ms": round(t_edge_ms, 1), "cloud_ms": round(t_cloud_ms, 1),
                    "edge_boxes": n_edge_boxes, "cloud_boxes": len(rc.boxes)})
    print(f"[{i+1}/{len(imgs)}] {img.name} conf={conf:.2f} edge={t_edge_ms*1000:.0f}ms"[:80])

Path("results/cache/cache.json").write_text(json.dumps(records, ensure_ascii=False, indent=1))

# ---------------- 阶段二：离线扫参（纯计算，秒级） ----------------
def summarize(records, route_fn):
    lat, up, flips = [], 0, 0
    for d in records:
        go_cloud = route_fn(d)
        total = d["edge_ms"] + (CLOUD_RTT_MS + d["cloud_ms"] if go_cloud else 0)
        lat.append(total)
        up += go_cloud
        # 云端复检是否改变了"是否检出缺陷"的结论
        detected_edge = d["conf_edge"] >= 0.5
        detected_final = (d["conf_cloud"] if go_cloud else d["conf_edge"]) >= 0.5
        flips += (go_cloud and detected_edge != detected_final)
    lat = np.array(lat)
    return {"上云比例": round(up / len(records), 3),
            "p50_ms": round(float(np.percentile(lat, 50)), 1),
            "p95_ms": round(float(np.percentile(lat, 95)), 1),
            "p99_ms": round(float(np.percentile(lat, 99)), 1),
            "复检翻转数": flips}

rows = []
# 对照组 A：全边缘（无样本满足 >= inf）
rows.append({"方案": "A_全边缘", **summarize(records, lambda d: False)})
# 对照组 B：全上云
rows.append({"方案": "B_全上云", **summarize(records, lambda d: True)})
# 置信度路由各档
for tau in TAU_LIST:
    rows.append({"方案": f"tau={tau}",
                 **summarize(records, lambda d, t=tau: d["conf_edge"] < t)})

# ---------------- 落盘与输出 ----------------
with open("results/tau_summary.csv", "w", newline="", encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys())
    w.writeheader()
    w.writerows(rows)

print(f"\n{'方案':<12} {'上云比例':<8} {'p50':<8} {'p95':<8} {'p99':<8} {'复检翻转':<8}")
for r in rows:
    print(f"{r['方案']:<12} {r['上云比例']:<8} {r['p50_ms']:<8} {r['p95_ms']:<8} "
          f"{r['p99_ms']:<8} {r['复检翻转数']:<8}")
print("\n已写入 results/tau_summary.csv 与 results/cache/cache.json（缓存供后续精度评估复用）")
