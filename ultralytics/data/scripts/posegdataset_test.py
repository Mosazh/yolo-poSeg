from ultralytics.data import PoSegDataset  # 替换为实际模块名
import yaml

with open("/home/Mos/Documents/Complex/MyStudy/new_yolo/yolo-poSeg/ultralytics/cfg/datasets/cocoTest-poseg.yaml", "r") as f:
    data = yaml.safe_load(f)

# 初始化数据集
dataset = PoSegDataset(
    img_path="/home/Mos/Documents/Complex/MyStudy/new_yolo/datasets/cocoTest-poseg/images",
    data=data,
    imgsz=640,
    augment=False
)
