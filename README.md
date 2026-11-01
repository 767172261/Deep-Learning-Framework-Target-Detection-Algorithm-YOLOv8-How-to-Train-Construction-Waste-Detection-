深度学习框架目标检测算法YOLOv8如何训练建筑垃圾检测数据集通过训练的权重推理识别检测砖头 混凝土瓷砖 木材 石材 塑料 管道 是高爆
@[toc]

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/21716e894fb74e0a97c040d29555e0b8.png)
1
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/eb700915fafe4071be2de6fb07d0322e.png)
1
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/3fbcf16d08ae4f30b109cc0179bd6913.png)
1
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/2cce8e7c5b09471eac030bba571465f1.png)

### 建筑废料目标检测数据集信息表

#### 数据集概述
| 项目 | 内容 |
|------|------|
| **数据集名称** | 建筑废料目标检测数据集 |
| **总图片数量** | 3,106 张 |
| **应用场景** | 建筑垃圾分类、智能回收管理、环境监测、建筑废料研究 |
| **标注格式** | txt（YOLO）、xml（Pascal VOC） |
| **类别数** | 10种建筑废料类别 |

#### 类别标签详情

| 序号 | 英文名称 | 中文名称 | 备注 |
|------|----------|----------|------|
| 0    | Brick    | 砖块/砖头 |      |
| 1    | Concrete | 混凝土   |      |
| 2    | Tile     | 瓷砖/瓦片 |      |
| 3    | Foam     | 泡沫材料 |      |
| 4    | General_w | 一般废弃物（综合废料） |      |
| 5    | Pipes    | 管道     |      |
| 6    | Plastic  | 塑料     |      |
| 7    | Gypsum_board | 石膏板 |      |
| 8    | Wood     | 木材     |      |
| 9    | Stone    | 石材/石料 |      |

#### 数据集划分

| 数据集类型 | 图片数量（占比） |
|------------|------------------|
| 训练集     | 2,174 张（70%）  |
| 验证集     | 621 张（20%）    |
| 测试集     | 311 张（10%）    |

使用 **YOLOv8** 进行训练和部署。


## 🛠️ 一、环境搭建（支持 GPU 加速）

### 1. 安装 NVIDIA 驱动和 CUDA

```bash
# 检查 GPU 是否可用
nvidia-smi
```

确保输出中显示您的显卡型号和 CUDA 版本（如 12.4）。若未安装，请前往：
👉 [https://www.nvidia.com/Download/index.aspx](https://www.nvidia.com/Download/index.aspx)

---

### 2. 安装 Anaconda

下载地址：[https://www.anaconda.com/products/distribution](https://www.anaconda.com/products/distribution)

安装后打开 **Anaconda Prompt**（Windows）或终端（Linux/macOS）

---

### 3. 创建 Python 虚拟环境

```bash
# 创建名为 waste_yolo 的环境
conda create -n waste_yolo python=3.10

# 激活环境
conda activate waste_yolo
```

---

### 4. 安装 PyTorch（支持 GPU）

推荐使用 CUDA 11.8 或 12.1：

```bash
# 示例：CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

验证是否成功：

```python
python -c "import torch; print(torch.cuda.is_available())"
# 应输出 True
```

---

### 5. 安装 YOLOv8 和其他依赖

```bash
# 安装 Ultralytics YOLOv8（官方最新版）
pip install ultralytics

# 安装 OpenCV 和可视化工具
pip install opencv-python opencv-python-headless matplotlib seaborn numpy pandas tqdm

# 日志与评估
pip install tensorboard scikit-learn
```

验证安装：

```bash
yolo version
# 输出应为：8.1.x 或更高
```

---

## 📁 二、数据集结构与 `data.yaml` 配置

### 文件结构（建议组织方式）

```
building_waste_dataset/
├── images/
│   ├── train/        # 3106 × 0.7 ≈ 2174 张
│   ├── val/          # 3106 × 0.2 ≈ 621 张
│   └── test/         # 3106 × 0.1 ≈ 311 张
├── labels/
│   ├── train/
│   ├── val/
│   └── test/
└── data.yaml
```

> ✅ 您已有划分，只需确保 `txt` 标注文件与图片同名且格式为 YOLO（归一化坐标）

---

### ✅ `data.yaml` 内容

```yaml
# data.yaml
train: ./building_waste_dataset/images/train
val: ./building_waste_dataset/images/val
test: ./building_waste_dataset/images/test  # YOLOv8.1+ 支持 test split

# 类别数量
nc: 10

# 类别名称（顺序必须与 label 中的 class_id 一致）
names: 
  - Brick
  - Concrete
  - Tile
  - Foam
  - General_w
  - Pipes
  - Plastic
  - Gypsum_board
  - Wood
  - Stone
```

> 💡 保存为 `building_waste_dataset/data.yaml`

---

## 🚀 三、训练代码：`train.py`

```python
# train.py
from ultralytics import YOLO
import torch

# ================== 配置参数 ==================
DATA_CONFIG = 'building_waste_dataset/data.yaml'     # 数据集配置文件
MODEL_NAME = 'yolov8m.pt'                            # 推荐使用 medium 模型（精度与速度平衡）
PROJECT = 'runs/waste_detection'                     # 项目保存路径
EXPERIMENT_NAME = 'yolov8m_building_waste'           # 实验名称
EPOCHS = 300                                         # 建筑废料建议训练充分
BATCH_SIZE = 32                                      # 根据显存调整（建议 ≥ 8GB GPU）
IMG_SIZE = 640                                       # 输入尺寸（可尝试 1280 提升小目标性能）
DEVICE = 0 if torch.cuda.is_available() else 'cpu'

# ================ 加载预训练模型 ================
model = YOLO(MODEL_NAME)

# ================ 开始训练 ================
results = model.train(
    data=DATA_CONFIG,
    epochs=EPOCHS,
    batch=BATCH_SIZE,
    imgsz=IMG_SIZE,
    project=PROJECT,
    name=EXPERIMENT_NAME,
    exist_ok=True,
    device=DEVICE,
    workers=8,
    optimizer='AdamW',              # 对复杂场景更稳定
    lr0=0.001,                      # 初始学习率
    lrf=0.1,                        # 最终学习率 = lr0 * lrf
    momentum=0.937,
    weight_decay=0.0005,
    patience=50,                    # 早停：50 轮无提升则停止
    save=True,
    save_period=10,
    cache=False,                    # 若内存充足可设为 True
    single_cls=False,
    close_mosaic=20,                # 最后 20 轮关闭 Mosaic
    augment=True,
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    degrees=10.0,
    translate=0.2,
    scale=0.5,
    shear=0.0,
    flipud=0.0,
    fliplr=0.5
)

print("✅ 训练完成！")
print(f"最佳模型保存路径：{PROJECT}/{EXPERIMENT_NAME}/weights/best.pt")
```

> 📌 训练完成后，最佳模型保存在：
> ```
> runs/waste_detection/yolov8m_building_waste/weights/best.pt
> ```

---

## 🔍 四、推理代码：`detect.py`

```python
# detect.py
from ultralytics import YOLO
import cv2
import os

# ================== 配置参数 ==================
MODEL_PATH = 'runs/waste_detection/yolov8m_building_waste/weights/best.pt'
SOURCE = 'demo_construction_site.mp4'  # 可为图片、视频、摄像头、文件夹
CONF_THRESHOLD = 0.25                  # 建筑废料建议低阈值以提高召回
IOU_THRESHOLD = 0.45
SHOW_RESULT = True
SAVE_RESULT = True
OUTPUT_DIR = 'runs/detect/waste_output'

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ================ 加载模型 ================
model = YOLO(MODEL_PATH)

# ================ 推理 ================
results = model.predict(
    source=SOURCE,
    conf=CONF_THRESHOLD,
    iou=IOU_THRESHOLD,
    save=SAVE_RESULT,
    project=OUTPUT_DIR,
    name='detection',
    exist_ok=True,
    imgsz=640,
    device=DEVICE,
    show=SHOW_RESULT,
    stream=True
)

# ================ 遍历结果 ================
for r in results:
    im_array = r.plot()
    if SHOW_RESULT:
        cv2.imshow('Building Waste Detection', im_array)
        if cv2.waitKey(1) == ord('q'):
            break

cv2.destroyAllWindows()
print(f"✅ 推理完成！结果保存在: {OUTPUT_DIR}")
```

> ✅ 支持输入类型：
> - 图片：`'site_001.jpg'`
> - 视频：`'excavator_footage.mp4'`
> - 摄像头：`0`
> - RTSP 流：`'rtsp://...'`

---

## 📊 五、评估代码：`val.py`（支持 test 集）

```python
# val.py
from ultralytics import YOLO

# ================== 配置参数 ==================
MODEL_PATH = 'runs/waste_detection/yolov8m_building_waste/weights/best.pt'
DATA_CONFIG = 'building_waste_dataset/data.yaml'
SPLIT = 'test'                     # 评估测试集
IMG_SIZE = 640
BATCH_SIZE = 32
CONF_THRESHOLD = 0.001
DEVICE = 0 if torch.cuda.is_available() else 'cpu'

# ================ 加载模型 ================
model = YOLO(MODEL_PATH)

# ================ 评估 ================
metrics = model.val(
    data=DATA_CONFIG,
    split=SPLIT,
    imgsz=IMG_SIZE,
    batch=BATCH_SIZE,
    conf=CONF_THRESHOLD,
    iou=0.6,
    device=DEVICE,
    plots=True,           # 生成 PR 曲线、混淆矩阵等
    save_json=True        # 生成 predictions.json
)

# ================ 打印指标 ================
print("\n" + "="*60)
print("📊 建筑废料目标检测模型评估结果（测试集）")
print("="*60)
print(f"mAP@0.5:        {metrics.box.map50:.4f}")
print(f"mAP@0.5:0.95:   {metrics.box.map:.4f}")
print(f"Precision:      {metrics.box.precision.mean():.4f}")
print(f"Recall:         {metrics.box.recall.mean():.4f}")
print(f"F1 Score:       {metrics.box.f1.mean():.4f}")

# 按类别输出 AP
class_names = ['Brick', 'Concrete', 'Tile', 'Foam', 'General_w', 'Pipes', 'Plastic', 'Gypsum_board', 'Wood', 'Stone']
for i, name in enumerate(class_names):
    print(f"{name:12} AP50: {metrics.box.ap[i]:.4f}")

print("="*60)
```

> 📈 输出图表：
> - `PR_curve.png`（可分析各类别召回率）
> - `confusion_matrix.png`（分析误检情况）
> - `results.png`（loss 和 mAP 曲线）

---

## 📈 六、性能优化建议

| 问题 | 建议 |
|------|------|
| 类别多（10类） | 使用 `yolov8m/l` 提升特征提取能力 |
| 小目标（碎片） | 使用 `imgsz=1280` 或 `multi-scale` 训练 |
| 类别不平衡 | 使用 `class weights` 或 `Focal Loss`（YOLOv8 默认启用） |
| 工业部署 | 导出为 ONNX/TensorRT 用于回收机器人 |


### 应用方向：
- 智能回收机器人（机械臂抓取定位）
- 建筑工地废料自动分类系统
- 无人机巡检 + 废料分布热力图生成
- 环境污染评估平台



![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/81edc92ec1584d96b4b0078d8f98e8f4.png)

