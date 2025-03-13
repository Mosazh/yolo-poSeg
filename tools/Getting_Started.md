
[issues_6949](https://github.com/ultralytics/ultralytics/issues/6949)
[Yolov8 项目结构](https://blog.csdn.net/qq_42452134/article/details/135151827)

以下是YOLO系列模型评估指标的详细解析：

### **指标全称与含义**
| 缩写        | 全称                          | 计算公式                           | 意义说明                                                                 |
|-------------|------------------------------|----------------------------------|--------------------------------------------------------------------------|
| **Box(P)**  | Bounding Box Precision        | TP / (TP + FP)                  | 检测框精确率：预测为正样本中实际为正的比例                                         |
| **Box(R)**  | Bounding Box Recall           | TP / (TP + FN)                  | 检测框召回率：实际正样本中被正确预测的比例                                         |
| **Box(mAP50)** | mean Average Precision@IoU=0.5 | 按类别计算AP后取平均                | 检测任务在IoU=0.5阈值下的平均精度（宽松标准）                                      |
| **Box(mAP50-95)** | COCO mAP                  | AP@IoU=0.5:0.05:0.95的平均值       | 检测任务在IoU=0.5~0.95（步长0.05）的平均精度（严格标准）                            |

| **Mask(P)** | Mask Precision                | 像素级TP / (像素级TP + FP)         | 分割掩模的像素级精确率                                                          |
| **Mask(R)** | Mask Recall                   | 像素级TP / (像素级TP + FN)         | 分割掩模的像素级召回率                                                          |
| **Mask(mAP50)** | Segmentation mAP@0.5       | 掩模IoU≥0.5时的平均精度              | 分割任务在IoU=0.5阈值下的平均精度                                                |
| **Mask(mAP50-95)** | COCO Mask mAP             | 掩模IoU=0.5~0.95的平均精度           | 分割任务在严格IoU阈值下的平均精度                                                |

| **Pose(P)** | Keypoints Precision           | 关键点预测正确的比例                  | 关键点坐标与真实值的距离≤阈值的比例                                                |
| **Pose(R)** | Keypoints Recall              | 被检测到的关键点占比                  | 实际存在的关键点中被正确预测的比例                                                 |
| **Pose(mAP50)** | Pose mAP@0.5                | 使用OKS（Object Keypoint Similarity）计算的AP | 关键点相似度≥0.5时的平均精度（宽松标准）                                           |
| **Pose(mAP50-95)** | COCO Keypoints mAP         | OKS=0.5:0.05:0.95的平均AP            | 关键点相似度在0.5~0.95范围内的平均精度（严格标准）                                   |

---

### **关键概念深度解析**
#### **1. IoU（Intersection over Union）**
- **检测任务**：预测框与真实框的重叠度
  ```python
  def calculate_iou(box1, box2):
      # box = [x1,y1,x2,y2]
      inter_area = max(0, min(box1[2],box2[2]) - max(box1[0],box2[0])) *
                   max(0, min(box1[3],box2[3]) - max(box1[1],box2[1]))
      union_area = (box1[2]-box1[0])*(box1[3]-box1[1]) +
                   (box2[2]-box2[0])*(box2[3]-box2[1]) - inter_area
      return inter_area / union_area
  ```

#### **2. OKS（Object Keypoint Similarity）**
姿态估计专用相似度度量，公式为：
```math
OKS = \frac{\sum_i exp(-d_i^2/(2s^2κ_i^2))δ(v_i>0)}{\sum_i δ(v_i>0)}
```
- $d_i$：第i个关键点的预测与真实坐标的欧氏距离
- $s$：目标尺度（√(width*height)）
- $κ_i$：每个关键点的归一化因子（COCO数据集预设）

---

### **指标解读示例**
假设某模型在验证集上的结果为：
```
Class     Images  Instances      Box(P    R   mAP50  mAP50-95)    Mask(P    R   mAP50  mAP50-95)    Pose(P    R   mAP50  mAP50-95)
all        1000     7563        0.892 0.864   0.901      0.672      0.832 0.811   0.843      0.612      0.781 0.752   0.792      0.583
```

#### **检测任务分析**
- **精确率0.892**：每100个预测框中有89个是真实的
- **召回率0.864**：模型能检测到86.4%的真实目标
- **mAP50-95为0.672**：在严格标准下，模型综合精度达到67.2%（COCO竞赛核心指标）

#### **分割任务短板**
- **Mask mAP50-95 (0.612) < Box mAP50-95 (0.672)**：说明掩模边缘精度不足
- **改进方向**：增加边缘敏感的数据增强（如弹性变换）

#### **姿态估计瓶颈**
- **Pose mAP50-95最低(0.583)**：关键点定位精度需提升
- **优化建议**：添加关键点热图监督或使用HRNet改进关键点预测头

---

### **各指标优先级排序**
| 指标类型       | 研发阶段优先级 | 部署阶段关注度 | 说明                     |
|----------------|----------------|----------------|--------------------------|
| mAP50-95       | ★★★★★         | ★★★★☆          | 综合性能的核心指标         |
| Recall         | ★★★★☆          | ★★★☆☆          | 漏检率的重要反映           |
| Precision      | ★★★☆☆          | ★★★★☆          | 误报率的关键指标           |
| 任务特定指标   | ★★★★☆          | ★★★★★          | 如自动驾驶更关注Recall    |

---

### **指标提升策略**
#### **提升检测mAP50-95**
1. 增加高分辨率训练（`imgsz=1280`）
2. 使用Albumentations进行几何增强
3. 添加[Task-Specific Context](https://arxiv.org/abs/2210.03160)模块

#### **优化分割Mask指标**
```python
# 在train参数中添加：
model.train(
    ...
    overlap_mask=True,  # 启用掩模重叠优化
    mask_ratio=4,       # 提高掩模预测头分辨率
    box=0.7,            # 调整检测与分割损失权重
    seg=0.3
)
```

#### **改善姿态估计Pose mAP**
1. 使用基于热图的姿态损失：
   ```yaml
   # poseg.yaml
   kpt_shape: [17, 3]          # 17个关键点，3表示(x,y,visibility)
   heatmap_loss: True          # 启用热图监督
   heatmap_size: [96, 96]      # 热图分辨率
   ```
2. 增加关键点可见性预测分支

通过理解这些指标的含义和相互关系，可以更精准地诊断模型缺陷并制定优化策略。

## 原始 yolov8n-pose 结构
``` text
WARNING ⚠️ no model scale passed. Assuming scale='n'.

                   from  n    params  module                                       arguments
  0                  -1  1       464  ultralytics.nn.modules.conv.Conv             [3, 16, 3, 2]
  1                  -1  1      4672  ultralytics.nn.modules.conv.Conv             [16, 32, 3, 2]
  2                  -1  1      7360  ultralytics.nn.modules.block.C2f             [32, 32, 1, True]
  3                  -1  1     18560  ultralytics.nn.modules.conv.Conv             [32, 64, 3, 2]
  4                  -1  2     49664  ultralytics.nn.modules.block.C2f             [64, 64, 2, True]
  5                  -1  1     73984  ultralytics.nn.modules.conv.Conv             [64, 128, 3, 2]
  6                  -1  2    197632  ultralytics.nn.modules.block.C2f             [128, 128, 2, True]
  7                  -1  1    295424  ultralytics.nn.modules.conv.Conv             [128, 256, 3, 2]
  8                  -1  1    460288  ultralytics.nn.modules.block.C2f             [256, 256, 1, True]
  9                  -1  1    164608  ultralytics.nn.modules.block.SPPF            [256, 256, 5]
 10                  -1  1         0  torch.nn.modules.upsampling.Upsample         [None, 2, 'nearest']
 11             [-1, 6]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 12                  -1  1    148224  ultralytics.nn.modules.block.C2f             [384, 128, 1]
 13                  -1  1         0  torch.nn.modules.upsampling.Upsample         [None, 2, 'nearest']
 14             [-1, 4]  1         0  ultralytics.nn.modules.conv.Concat           [1]
 15                  -1  1     37248  ultralytics.nn.modules.block.C2f             [192, 64, 1]
 16                  -1  1     36992  ultralytics.nn.modules.conv.Conv             [64, 64, 3, 2]
 17            [-1, 12]  1         0  ultralytics.nn.modules.conv.Concat           [1]
 18                  -1  1    123648  ultralytics.nn.modules.block.C2f             [192, 128, 1]
 19                  -1  1    147712  ultralytics.nn.modules.conv.Conv             [128, 128, 3, 2]
 20             [-1, 9]  1         0  ultralytics.nn.modules.conv.Concat           [1]
 21                  -1  1    493056  ultralytics.nn.modules.block.C2f             [384, 256, 1]
 22        [15, 18, 21]  1   1035934  ultralytics.nn.modules.head.Pose             [1, [17, 3], [64, 128, 256]]
YOLOv8-pose summary: 144 layers, 3,295,470 parameters, 3,295,454 gradients, 9.3 GFLOPs
```

## 原始 yolov8n-seg 模型结构
``` text
WARNING ⚠️ no model scale passed. Assuming scale='n'.

                   from  n    params  module                                       arguments
  0                  -1  1       464  ultralytics.nn.modules.conv.Conv             [3, 16, 3, 2]
  1                  -1  1      4672  ultralytics.nn.modules.conv.Conv             [16, 32, 3, 2]
  2                  -1  1      7360  ultralytics.nn.modules.block.C2f             [32, 32, 1, True]
  3                  -1  1     18560  ultralytics.nn.modules.conv.Conv             [32, 64, 3, 2]
  4                  -1  2     49664  ultralytics.nn.modules.block.C2f             [64, 64, 2, True]
  5                  -1  1     73984  ultralytics.nn.modules.conv.Conv             [64, 128, 3, 2]
  6                  -1  2    197632  ultralytics.nn.modules.block.C2f             [128, 128, 2, True]
  7                  -1  1    295424  ultralytics.nn.modules.conv.Conv             [128, 256, 3, 2]
  8                  -1  1    460288  ultralytics.nn.modules.block.C2f             [256, 256, 1, True]
  9                  -1  1    164608  ultralytics.nn.modules.block.SPPF            [256, 256, 5]
 10                  -1  1         0  torch.nn.modules.upsampling.Upsample         [None, 2, 'nearest']
 11             [-1, 6]  1         0  ultralytics.nn.modules.conv.Concat           [1]
 12                  -1  1    148224  ultralytics.nn.modules.block.C2f             [384, 128, 1]
 13                  -1  1         0  torch.nn.modules.upsampling.Upsample         [None, 2, 'nearest']
 14             [-1, 4]  1         0  ultralytics.nn.modules.conv.Concat           [1]
 15                  -1  1     37248  ultralytics.nn.modules.block.C2f             [192, 64, 1]
 16                  -1  1     36992  ultralytics.nn.modules.conv.Conv             [64, 64, 3, 2]
 17            [-1, 12]  1         0  ultralytics.nn.modules.conv.Concat           [1]
 18                  -1  1    123648  ultralytics.nn.modules.block.C2f             [192, 128, 1]
 19                  -1  1    147712  ultralytics.nn.modules.conv.Conv             [128, 128, 3, 2]
 20             [-1, 9]  1         0  ultralytics.nn.modules.conv.Concat           [1]
 21                  -1  1    493056  ultralytics.nn.modules.block.C2f             [384, 256, 1]
 22        [15, 18, 21]  1   1150432  ultralytics.nn.modules.head.Segment          [80, 32, 64, [64, 128, 256]]
YOLOv8-seg summary: 151 layers, 3,409,968 parameters, 3,409,952 gradients, 12.8 GFLOPs
```

## 我的模型结构 yolov8n-poseg (error)
``` text
WARNING ⚠️ no model scale passed. Assuming scale='n'.

                   from  n    params  module                                       arguments
  0                  -1  1       464  ultralytics.nn.modules.conv.Conv             [3, 16, 3, 2]
  1                  -1  1      4672  ultralytics.nn.modules.conv.Conv             [16, 32, 3, 2]
  2                  -1  1      7360  ultralytics.nn.modules.block.C2f             [32, 32, 1, True]
  3                  -1  1     18560  ultralytics.nn.modules.conv.Conv             [32, 64, 3, 2]
  4                  -1  2     49664  ultralytics.nn.modules.block.C2f             [64, 64, 2, True]
  5                  -1  1     73984  ultralytics.nn.modules.conv.Conv             [64, 128, 3, 2]
  6                  -1  2    197632  ultralytics.nn.modules.block.C2f             [128, 128, 2, True]
  7                  -1  1    295424  ultralytics.nn.modules.conv.Conv             [128, 256, 3, 2]
  8                  -1  1    460288  ultralytics.nn.modules.block.C2f             [256, 256, 1, True]
  9                  -1  1    164608  ultralytics.nn.modules.block.SPPF            [256, 256, 5]
 10                  -1  1         0  torch.nn.modules.upsampling.Upsample         [None, 2, 'nearest']
 11             [-1, 6]  1         0  ultralytics.nn.modules.conv.Concat           [1]
 12                  -1  1    148224  ultralytics.nn.modules.block.C2f             [384, 128, 1]
 13                  -1  1         0  torch.nn.modules.upsampling.Upsample         [None, 2, 'nearest']
 14             [-1, 4]  1         0  ultralytics.nn.modules.conv.Concat           [1]
 15                  -1  1     37248  ultralytics.nn.modules.block.C2f             [192, 64, 1]
 16                  -1  1     36992  ultralytics.nn.modules.conv.Conv             [64, 64, 3, 2]
 17            [-1, 12]  1         0  ultralytics.nn.modules.conv.Concat           [1]
 18                  -1  1    123648  ultralytics.nn.modules.block.C2f             [192, 128, 1]
 19                  -1  1    147712  ultralytics.nn.modules.conv.Conv             [128, 128, 3, 2]
 20             [-1, 9]  1         0  ultralytics.nn.modules.conv.Concat           [1]
 21                  -1  1    493056  ultralytics.nn.modules.block.C2f             [384, 256, 1]
 22        [15, 18, 21]  1   1002107  ultralytics.nn.modules.head.PoSeg            [80, 32, 64, [17, 3], [64, 128, 256]]
YOLOv8-poseg summary: 178 layers, 3,261,643 parameters, 3,261,627 gradients, 12.1 GFLOPs
```
