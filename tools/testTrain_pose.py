from ultralytics import YOLO

# Load a model
model = YOLO("yolov8-pose.yaml")  # build a new model from YAML

# Train the model
results = model.train(data="coco8-pose.yaml", epochs=100, imgsz=640)
