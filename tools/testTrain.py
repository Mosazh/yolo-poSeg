from ultralytics import YOLO


# Load a model
model = YOLO("yolov8-poseg.yaml")

# Train the model
# results = model.train(data="cocoTest-poseg.yaml", device='cuda', optimizer='Adam', epochs=100, imgsz=640, batch=32)
results = model.train(
    data="cocoTest-poseg.yaml",
    # device='0',
    optimizer='SGD',
    epochs=30,
    imgsz=640,
    batch=1,
    lr0=0.001,               # 初始学习率
    lrf=0.1,                # 余弦退火
    momentum=0.9,           # SGD动量
    weight_decay=1e-5,      # 调小权重衰减
    # warmup_epochs=3,        # Warmup阶段
    # patience=15,            # 延长早停
    # hsv_h=0.015,            # 增强参数
    # mixup=0.2               # 启用Mixup
)
