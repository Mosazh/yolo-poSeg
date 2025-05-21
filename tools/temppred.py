from ultralytics import YOLO

# Load a pretrained YOLO11n model
model = YOLO("/home/Mos/Documents/Complex/MyStudy/new_yolo/train_record/poseg/poSeg-mobilenetv4-v8_siou_iouSigma_0.05_box_30/weights/best.pt")

# Define path to the image file
source = "/home/Mos/Desktop/mtemp/complete_test/Experimental_plot_02_s2.png"

# Run inference on the source
results = model(source, imgsz=640, conf=0.4, iou=0.7, save_txt=True, save=True, show=True, device="cpu")  # list of Results objects
