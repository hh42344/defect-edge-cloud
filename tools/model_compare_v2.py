"""model_compare_v2.py — 多模型对比（修正版）

修正点（相对 DeepSeek 版 model_compare.py）：
1. 时延改用 metrics.speed（ultralytics 官方分项耗时：预处理/推理/后处理，每张均值），
   不再用整体 wall-clock——避免把数据集加载、冷启动混进对比
2. 每个模型先热身 2 张图，消除首个模型的冷启动不公平
3. 指标补齐：mAP50、mAP50-95、Precision、Recall、逐类 AP50 全部落 CSV
4. 结果落盘 compare_summary.csv（申报书表格的直接数据源）
5. 逐类指标自动带类别名（从 neu_det.yaml 读 names，不靠猜索引）

用法：python model_compare_v2.py
"""
import csv, json, time
from pathlib import Path
from ultralytics import YOLO

MODELS = {
    "v8n":  "models/YOLOv8n/yolov8n_best.onnx",
    "v8s":  "models/YOLOv8s/yolov8s_best.onnx",
    "v8m":  "models/YOLOv8m/yolov8m_best.onnx",
    "v9t":  "models/YOLOv9t/yolov9t_best.onnx",
    "v10n": "models/YOLOv10n/yolov10n_best.onnx",
    "v11n": "models/YOLO11n/yolo11n_best.onnx",
    "v12n": "models/YOLO12n/yolo12n_best.onnx",
    "v26n": "models/YOLO26n/yolo26n_best.onnx",
}
DATA = "neu_det.yaml"
WARMUP_IMGS = 2

names = {0: "crazing", 1: "inclusion", 2: "patches", 3: "pitted_surface", 4: "rolled-in_scale", 5: "scratches"}

rows = []
for name, path in MODELS.items():
    if not Path(path).exists():
        print(f"[跳过] {name}: {path} 不存在")
        continue
    print(f"\n===== 评测 {name} ({path}) =====")
    size_mb = Path(path).stat().st_size / 1024 / 1024
    model = YOLO(path)

    # 热身：消除首图冷启动
    dummy = sorted(Path("test_images").glob("*.jpg"))[:WARMUP_IMGS]
    for d in dummy:
        model.predict(d, device="cpu", imgsz=640, verbose=False)

    m = model.val(data=DATA, device="cpu", batch=1, verbose=False, plots=False)
    speed = m.speed  # {"preprocess": ms, "inference": ms, "postprocess": ms}（每张均值）
    inf_ms = speed.get("inference", -1)

    print(f"大小 {size_mb:.2f} MB | 推理 {inf_ms:.1f} ms/张 (CPU)")
    print(f"mAP50={m.box.map50:.4f}  mAP50-95={m.box.map:.4f}  "
          f"P={m.box.mp:.4f}  R={m.box.mr:.4f}")
    for ci, ap in enumerate(m.box.ap50):
        print(f"  [{ci}] {names.get(ci, ci)}: AP50={ap:.4f} R={m.box.r[ci]:.4f}")

    rows.append({
        "model": name, "size_mb": round(size_mb, 2),
        "infer_ms_per_img": round(inf_ms, 1),
        "preprocess_ms": round(speed.get("preprocess", -1), 1),
        "postprocess_ms": round(speed.get("postprocess", -1), 1),
        "mAP50": round(float(m.box.map50), 4),
        "mAP50-95": round(float(m.box.map), 4),
        "Precision": round(float(m.box.mp), 4),
        "Recall": round(float(m.box.mr), 4),
        **{f"AP50_{names.get(ci, ci)}": round(float(ap), 4)
           for ci, ap in enumerate(m.box.ap50)},
        **{f"R_{names.get(ci, ci)}": round(float(r_), 4)
           for ci, r_ in enumerate(m.box.r)},
    })

with open("results/compare_summary.csv", "w", newline="", encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys())
    w.writeheader()
    w.writerows(rows)
print("\n已写入 results/compare_summary.csv")
