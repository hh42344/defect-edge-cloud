import json, time
from pathlib import Path
import numpy as np
from ultralytics import YOLO

edge = YOLO("models/YOLOv8n/yolov8n_best.onnx")
cloud = YOLO("models/YOLO12n/yolo12n_best.pt")

TAU_ARRAY = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
CLOUD_RTT = 0.080   
images = sorted(Path("test_images").glob("*.jpg"))

print(f"{'TAU':<5} | {'上云比例':<8} | {'p50(ms)':<8} | {'p95(ms)':<8} | {'平均置信度':<10} | {'缺陷漏检数':<10}")
print("-" * 75)

all_results = []

for tau in TAU_ARRAY:
    lat = []
    cloud_count = 0
    total_conf = 0.0
    miss_count = 0  # 漏检数（即：边缘端直出，但置信度极低甚至为0的情况）
    
    for img in images:
        t0 = time.perf_counter()
        r = edge(img, device="cpu", conf=0.01, imgsz=640, verbose=False)[0]
        t_edge = time.perf_counter() - t0
        
        conf = float(r.boxes.conf.max()) if len(r.boxes) else 0.0
        
        if conf < tau:
            # 触发上云
            time.sleep(CLOUD_RTT)
            t1 = time.perf_counter()
            cloud_r = cloud(img, device="cpu", conf=0.01, imgsz=960, verbose=False)[0]
            t_cloud = time.perf_counter() - t1
            total = t_edge + CLOUD_RTT + t_cloud
            cloud_count += 1
            
            final_conf = float(cloud_r.boxes.conf.max()) if len(cloud_r.boxes) else 0.0
        else:
            total = t_edge
            final_conf = conf
            
            # 如果边缘端直出了，但置信度极低（比如低于0.3），我们认为这是一次极度危险的“高危漏检”
            if final_conf < 0.3:
                miss_count += 1
                
        total_conf += final_conf
        lat.append(total * 1000)
        
    cloud_ratio = cloud_count / len(images)
    p50 = np.percentile(lat, 50)
    p95 = np.percentile(lat, 95)
    avg_conf = total_conf / len(images)
    
    print(f"{tau:<5} | {cloud_ratio:<10.1%} | {p50:<8.0f} | {p95:<8.0f} | {avg_conf:<10.3f} | {miss_count:<10}")

    all_results.append({
        "tau": tau, "cloud_ratio": cloud_ratio, "p50": p50, "p95": p95, 
        "avg_conf": avg_conf, "miss_count": miss_count
    })