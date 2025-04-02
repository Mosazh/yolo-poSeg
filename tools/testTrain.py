from ultralytics import YOLO

def main():
    # Load a model
    model = YOLO("yolov8-poseg.yaml")

    # Train the model
    # results = model.train(data="cocoTest-poseg.yaml", device='cuda', optimizer='Adam', epochs=100, imgsz=640, batch=32)
    results = model.train(
        data="MKSD.yaml",
        epochs=2,
        imgsz=640,
        batch=1,
        hsv_h=0.0,    # 关闭色调增强
        hsv_s=0.0,    # 关闭饱和度增强
        hsv_v=0.0,    # 关闭亮度增强
        degrees=0.0,  # 关闭旋转增强
        translate=0.0,  # 关闭平移增强
        scale=0.0,    # 关闭缩放增强
        shear=0.0,    # 关闭剪切增强
        perspective=0.0,  # 关闭透视变换
        flipud=0.0,   # 关闭上下翻转
        fliplr=0.0,   # 关闭左右翻转
        mosaic=0.0,   # 关闭Mosaic增强
        mixup=0.0     # 关闭MixUp增强
    )

if __name__ == '__main__':

    main()
