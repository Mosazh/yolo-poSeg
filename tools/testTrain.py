from ultralytics import YOLO


# Load a model
model = YOLO("yolov8-poseg.yaml")

# Train the model
results = model.train(data="cocoTest-poseg.yaml", epochs=100, imgsz=640, batch=32)
