# 面向产线视觉质检的云边协同 AI 服务编排系统

NEU-DET 钢材表面缺陷质检（1800 张，6 类，200×200 灰度）。

**架构一句话**：边缘端轻量检测（YOLOv10n）负责"**别漏**"，云端 ROI 分类复检（ResNet）负责"**别乱报**"，中间用学习型路由分配上云预算。

---

## 当前阶段

**阶段 2 · 云端复检 + 路由**（2026-11-21 起，预计 12/10 收尾）

目标：把"边缘低阈值多报 → 路由挑 k 个 → 云端 ROI 分类确认/驳回"这条链跑通，并证明它在同等上云预算下召回更高。

---

## 目录结构

```
defect-edge-cloud/
├── neu_det.yaml          ★ 数据集配置，所有脚本都引用它，不要移动
├── README.md
├── .gitignore
│
├── tools/                  所有代码（5 个脚本）
├── results/                所有数据表（csv / json）
│   └── cache/              逐图 / 逐框推理缓存
├── figures/                所有图片
├── docs/                   文档
├── other/                  早期版本归档（v1 的脚本与日志，不再使用）
│
├── images/  labels/        数据集（train 1440 / val 360）
├── models/                 各模型权重与 onnx
├── runs/                   训练日志（ultralytics 自动生成）
└── test_images/            热身用的 19 张图
```

---

## 快速开始

> ⚠️ **所有命令必须在项目根目录 `D:\Projects\defect-edge-cloud` 下执行。**
> 脚本里用的是相对路径（`images/`、`neu_det.yaml`），在 `tools/` 里直接跑会全部找不到。

```powershell
cd D:\Projects\defect-edge-cloud
conda activate <你的环境名>
```

### 1. GT 框尺寸统计（约 10 秒）

```powershell
python tools\box_stats.py
```

产出：`results/box_stats.csv`

### 2. 逐框评估 —— Recall / FPPI / FROC（约 2 分钟）

```powershell
python tools\eval_metrics.py
python tools\eval_metrics.py --model models\YOLOv10n\yolov10n_best.onnx --split val
```

产出：

- `results/cache/cache_v2.json` —— **逐框**推理结果（路由的燃料）
- `results/froc.csv` —— 95 个阈值下的 TP/FP/FN/Recall/Precision/FPPI
- `results/metrics_by_class.csv` —— 分类别指标
- `figures/fig_froc.png`

### 3. 路由参考基线 —— 预算 vs 召回（约 5 秒）

依赖第 2 步产出的 `cache_v2.json`，**不重跑推理**。

```powershell
python tools\plot_baseline.py
```

产出：

- `results/baseline.csv` —— 每档 k × 四种策略（random / conf_asc / conf_desc / oracle）
- `figures/fig_baseline.png` —— 四条基线曲线

> ⚠️ 当前云端判决由 GT 充当（oracle 假设），曲线为**理论上界**。分类器训练完成后替换判决逻辑即为实测曲线。
> 详见 [docs/实验_路由基线.md](docs/实验_路由基线.md)

### 4. 阈值扫参（约 15 分钟）

```powershell
python tools\route_v2.py
```

产出：`results/tau_summary.csv`、`results/cache/cache.json`

### 5. 八模型横评（约 1 小时）

```powershell
python tools\model_compare_v2.py
```

产出：`results/compare_summary.csv`

### 6. 出图

```powershell
python tools\plot_results.py
```

产出：`figures/fig_tau_curve.png`、`figures/fig_model_compare.png`

---

## 仓库约定（Git）

### 什么进仓库

```
✅ 代码        tools/*.py
✅ 文档        docs/、README.md
✅ 结果表      results/*.csv
✅ 论文图      figures/*.png
✅ 标签        labels/**/*.txt（1800 个小文件，0.5MB）
✅ 配置        neu_det.yaml
✅ 归档        other/
✅ 训练日志    models/**/results.csv、models/**/args.yaml（几十 KB）
```

### 什么不进仓库

```
❌ .conda_env/          1.2GB 的 Python 环境
❌ *.pt / *.onnx        模型权重（几十到几百 MB）
❌ images/              数据集原图（25MB）
❌ runs/                训练过程文件（135MB）
❌ results/cache/*.json 推理缓存（能重跑）
❌ 新建文件夹/           别人论文的复现代码
```

### 为什么不进

**GitHub 对单个文件有 100MB 硬限制**，超了直接拒绝推送。
而且权重和原图是"二进制大文件"，放进 git 会让仓库越滚越大、clone 越来越慢。

### 克隆之后你缺什么

```powershell
git clone <仓库地址>
```

拿到的是**代码 + 文档 + 结果表**，但**跑不起来**，还需要自己准备：

| 缺什么 | 从哪来 |
|---|---|
| **数据集原图** `images/` | 向队长要（网盘），放到项目根目录 |
| **模型权重** `models/**/*.onnx` | 同上 |
| **Python 环境** | 自己 `conda create` + `pip install ultralytics` |

**标签 `labels/` 在仓库里**，不用另外要。

---

## 当前的关键数字（2026-11-22）

边缘模型 v10n，验证集 360 张 / 838 个真实缺陷框：

| 工作点 | Recall | FPPI | Precision |
|---|---|---|---|
| conf ≥ 0.05 | 0.865 | 2.60 | 0.436 |
| conf ≥ 0.15 | 0.794 | 1.21 | 0.605 |
| conf ≥ 0.30 | 0.703 | 0.61 | 0.729 |
| conf ≥ 0.50 | 0.603 | 0.23 | 0.857 |

**单类短板**：crazing 的 Recall 只有 **0.329**（140 个真缺陷漏掉 94 个），其余五类 ≥ 0.73。

**系统召回天花板**：即使把候选池阈值降到 0.05，仍有 13.5% 的缺陷连框都没有 —— **路由救不回来**。

---

## 五条硬约束

1. **必须从项目根目录运行脚本**，否则相对路径全错。
2. **`neu_det.yaml` 不要移动或改名**，4 个脚本都按这个相对路径找它。
3. **训练集 / 验证集必须按"图"划分**，不能按"框"划分（同一张图的框跨集会数据泄露）。
4. **路由特征不得使用云端信息**（`conf_cloud`、云端类别、与 GT 的 IoU）——那是决策之后才有的，用了就是作弊。
5. **只报 Recall 和 FPPI，不报总准确率** —— 数据严重不平衡，总准确率会骗人。

---

## 其他

- `other/` 是 v1 早期探索的归档（19 张图、图级路由、未热身），**已被 `tools/route_v2.py` 取代，仅作历史存证，不要用它跑实验**。
- `tools/` 下的 `*.bak` 是整理目录时留的备份，验证通过后可删。
