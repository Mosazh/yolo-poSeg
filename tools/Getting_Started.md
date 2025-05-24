
[issues_6949](https://github.com/ultralytics/ultralytics/issues/6949)
[Yolov8 项目结构](https://blog.csdn.net/qq_42452134/article/details/135151827)

## Original yolov8n-pose architecture
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

## Original yolov8n-seg architecture
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

## Base yolov8n-poseg architecture
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
 22        [15, 18, 21]  1   1446037  ultralytics.nn.modules.head.PoSeg            [2, 32, 64, [1, 3], [64, 128, 256]]
YOLOv8-poseg summary: 181 layers, 3,705,573 parameters, 3,705,557 gradients, 12.4 GFLOPs
```

## Metrics
在深度学习中，召回率（**Recall**）和精度（**Precision**）是两个核心评估指标，它们分别衡量模型在不同方面的性能，但通常存在此消彼长的关系（称为 **Precision-Recall Tradeoff**）。以下是它们的定义、关系和实际应用分析：

---

### **1. 定义与公式**
#### **(1) 召回率（Recall）**
• **目标**：模型能正确识别所有正样本中的多少比例。
• **公式**：
  \[
  \text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}
  \]
  • **TP（True Positive）**：正确预测的正样本。
  • **FN（False Negative）**：漏检的正样本。
• **意义**：召回率高，表示模型“宁可错检，不可漏检”。

#### **(2) 精度（Precision）**
• **目标**：模型预测的正样本中有多少是真正的正样本。
• **公式**：
  \[
  \text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}
  \]
  • **FP（False Positive）**：误检的负样本。
• **意义**：精度高，表示模型“宁可漏检，不可错检”。

---

### **2. 召回率与精度的关系**
#### **(1) 此消彼长（Tradeoff）**
• **原因**：调整分类阈值（如目标检测中的置信度阈值）会直接影响两者：
  • **提高阈值** → 模型更保守 → **精度↑，召回率↓**（减少误检，但漏检增多）。
  • **降低阈值** → 模型更激进 → **召回率↑，精度↓**（减少漏检，但误检增多）。
• **示例**：
  • **阈值=0.9**：仅保留高置信度预测 → 精度高，召回率低。
  • **阈值=0.1**：接受低置信度预测 → 召回率高，精度低。

召回率（Recall）和精度（Precision）是评估分类模型性能的两个重要指标，它们描述了模型的不同方面。确实，在很多情况下，提高其中一个指标可能会导致另一个指标下降，这是因为两者之间往往存在一种权衡关系。

- **召回率**指的是在所有实际为正类的样本中，模型正确识别出来的比例。高召回率意味着模型能够识别出大多数的正类实例，但同时也可能误判一些负类实例为正类。
- **精度**指的是在所有被模型预测为正类的样本中，实际上确实是正类的比例。高精度意味着在模型预测为正类的结果中有很高的可信度，但可能会漏掉一些实际上为正类的样本。

这种权衡关系是因为：

1. 如果一个模型倾向于将更多的样本预测为正类，那么它有可能捕捉到更多的真正正类样本（提高召回率），但这也会增加错误地将负类样本标记为正类的可能性（降低精度）。
2. 相反，如果一个模型非常保守，只在非常确定的情况下才预测为正类，这样可以确保预测为正类的样本有很高的准确性（提高精度），但是会漏掉许多实际上是正类的样本（降低召回率）。

然而，并不是所有的场景下都必然存在这种权衡。通过改进模型、特征工程或者使用更复杂的技术（如集成学习方法），可以在某些情况下同时提高召回率和精度。此外，调整决策阈值也可以帮助找到召回率和精度之间的最佳平衡点。最终目标是在特定应用场景下，根据业务需求找到最合适的平衡点。例如，在某些安全敏感的应用中，可能更注重高召回率以确保尽可能多的实际正例被捕捉，而在其他应用中则可能优先考虑高精度。

#### **(2) 可视化：PR曲线**
• **PR曲线（Precision-Recall Curve）**：通过不同阈值下的精度和召回率绘制曲线。
• **曲线下面积（AP, Average Precision）**：衡量模型整体性能（AP越高，模型越好）。

![](https://miro.medium.com/v2/resize:fit:1400/1*4so5DlCiwYg1gB6hYoLxJQ.png)

---

### **3. 实际应用中的权衡**
#### **(1) 不同场景的需求**
• **高召回率优先**：
  • **应用场景**：医疗诊断（如癌症筛查）、安全检测（如机场安检）。
  • **代价**：允许一定的误检（FP），但需减少漏检（FN）。
• **高精度优先**：
  • **应用场景**：垃圾邮件分类、商品推荐。
  • **代价**：允许少量漏检（FN），但需避免误检（FP）。

#### **(2) 平衡策略**
• **调整分类阈值**：根据业务需求选择合适的阈值（例如目标检测中通过验证集确定最佳阈值）。
• **优化损失函数**：使用 **F1 Score**、**Fβ Score** 或自定义损失函数平衡二者：
  \[
  \text{F1} = \frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}, \quad
  \text{Fβ} = (1 + \beta^2) \cdot \frac{\text{Precision} \cdot \text{Recall}}{\beta^2 \cdot \text{Precision} + \text{Recall}}
  \]
  • **β>1**：更重视召回率（如β=2）。
  • **β<1**：更重视精度（如β=0.5）。

---

### **4. 在目标检测中的特殊表现**
在目标检测任务（如YOLO）中，召回率和精度的计算与分类任务类似，但需结合IoU（交并比）和置信度：
1. **IoU阈值**：预测框与真实框的IoU超过阈值（如0.5）才视为TP。
2. **置信度阈值**：过滤低置信度的预测框。
3. **NMS（非极大值抑制）**：进一步合并重叠预测框，可能影响召回率。

#### **优化策略**
• **数据层面**：
  • 增加小目标样本数量（提升召回率）。
  • 平衡正负样本分布（减少FP）。
• **模型层面**：
  • 调整损失函数权重（如提高边界框损失权重）。
  • 使用更鲁棒的回归损失（如EIoU代替IoU）。
• **后处理**：
  • 动态调整NMS参数（如`iou_threshold`和`score_threshold`）。

---

### **5. 案例分析**
#### **(1) 漏检严重（低召回率）**
• **现象**：模型漏检大量目标（如小目标或遮挡目标）。
• **解决方法**：
  • 降低分类阈值或NMS的IoU阈值。
  • 在损失函数中增加正样本权重（如Focal Loss）。

#### **(2) 误检过多（低精度）**
• **现象**：模型频繁误检背景或负样本。
• **解决方法**：
  • 提高分类阈值或置信度过滤阈值。
  • 增加难负样本挖掘（Hard Negative Mining）。

---

### **6. 总结**
召回率和精度是模型性能的两个关键视角，理解它们的权衡关系是优化模型的核心。实际应用中需结合具体场景选择策略，并通过实验验证（如PR曲线、混淆矩阵）不断迭代模型。
