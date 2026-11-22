import os
import time
import numpy as np
from ultralytics import YOLO

# 把学弟发给你的模型路径放在这里
models_to_test = {
    "V8n": "models/YOLOv8n/yolov8n_best.onnx", 
    "V8s": "models/YOLOv8s/yolov8s_best.onnx",
    "V8m": "models/YOLOv8m/yolov8m_best.onnx",  
    "V11n": "models/YOLO11n/yolo11n_best.onnx",
    "V12n": "models/YOLO12n/yolo12n_best.onnx",
}

test_images = "test_images"

for name, model_path in models_to_test.items():
    print(f"\n================ 开始评测: {name} ================")
    
    # 1. 测文件大小
    size_mb = os.path.getsize(model_path) / (1024 * 1024)
    print(f"📦 模型大小: {size_mb:.2f} MB")
    
    model = YOLO(model_path)
    
    # 2. 测速度
    start = time.perf_counter()
    # 跑验证集，只看速度不看画图
    metrics = model.val(data="neu_det.yaml", device="cpu", batch=1, verbose=False, plots=False)
    total_time = time.perf_counter() - start
    print(f"⏱️ 验证耗时: {total_time:.2f} s (包含后处理)")
    
    # 3. 看核心指标 (mAP 和 Recall)
    print(f"🎯 整体 mAP50: {metrics.box.map50:.3f}")
    print(f"🕵️ 整体 Recall: {metrics.box.mr:.3f}")
    
    # 4. 重点看微裂纹 (假设 crazing 的索引是 0)
    # 注意：你需要根据 yaml 里的 names 来确认索引
    print(f"🔍 微裂纹 Recall: {metrics.box.r[0]:.3f} (越低说明漏检越严重)")