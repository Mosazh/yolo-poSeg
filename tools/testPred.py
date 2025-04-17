from ultralytics import YOLO
import pandas as pd
import os
import cv2

os.environ['QT_QPA_PLATFORM'] = 'xcb'
os.chdir(os.path.dirname(__file__))

model = YOLO("/home/Mos/Documents/Complex/MyStudy/new_yolo/yolo-poSeg/runs/train3/weights/best.pt")

results = model.predict(
    source="/home/Mos/Desktop/mtemp/complete_test/Experimental_plot_01-03.png",  # 输入图像路径
    conf=0.25,         # 置信度阈值（过滤低置信度检测）
    iou=0.7,           # IoU 阈值（用于NMS）
    imgsz=640,         # 输入图像尺寸（自动缩放）
    save=True,         # 保存结果图像
    show=True,         # 显示结果图像
    save_txt=True,     # 保存检测结果为txt文件（YOLO格式）
    device="cpu"    # 使用GPU（可选：'cpu' 或 'cuda:0'）
)

# 遍历每一帧的检测结果
for result in results:
    # 获取检测框信息
    boxes = result.boxes.xyxy   # 边界框坐标 (x1, y1, x2, y2)
    confs = result.boxes.conf   # 置信度
    cls_ids = result.boxes.cls  # 类别ID

    # 获取类别名称映射
    names = model.names

    # 遍历每个检测框
    for box, conf, cls_id in zip(boxes, confs, cls_ids):
        x1, y1, x2, y2 = box.tolist()
        label = names[int(cls_id)]  # 类别名称（如 'person'）
        print(f"检测到 {label}, 置信度: {conf:.2f}, 位置: [{x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f}]")

# 提取所有检测结果到 DataFrame
data = []
for result in results:
    for box, conf, cls_id in zip(result.boxes.xyxy, result.boxes.conf, result.boxes.cls):
        data.append({
            "class": model.names[int(cls_id)],
            "confidence": float(conf),
            "x1": float(box[0]),
            "y1": float(box[1]),
            "x2": float(box[2]),
            "y2": float(box[3])
        })

# 保存为 CSV
df = pd.DataFrame(data)
df.to_csv("detections.csv", index=False)
