from ultralytics import YOLO
import cv2
import numpy as np
import pandas as pd
import os
from tqdm import tqdm

class SlidingWindowPredictor:
    def __init__(self, model_path, window_size=640, overlap=0.2):
        self.model = YOLO(model_path)
        self.window_size = window_size
        self.stride = int(window_size * (1 - overlap))  # 滑动步长
        self.results_df = pd.DataFrame(columns=["x1", "y1", "x2", "y2", "conf", "cls", "window_pos"])

    def _sliding_windows(self, img):
        """生成滑动窗口坐标"""
        h, w = img.shape[:2]
        windows = []
        for y in range(0, h, self.stride):
            for x in range(0, w, self.stride):
                x_end = min(x + self.window_size, w)
                y_end = min(y + self.window_size, h)
                windows.append((x, y, x_end, y_end))
        return windows

    def predict_and_save(self, image_path, output_dir):
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(image_path))[0]

        # 读取原始图像
        img = cv2.imread(image_path)
        full_h, full_w = img.shape[:2]
        merged_img = np.zeros_like(img)  # 用于拼接预测结果
        mask = np.zeros((full_h, full_w), dtype=np.uint8)  # 记录覆盖区域

        # 滑动窗口预测
        windows = self._sliding_windows(img)
        for idx, (x1, y1, x2, y2) in enumerate(tqdm(windows)):
            # 裁剪窗口区域
            crop = img[y1:y2, x1:x2]

            # YOLOv8预测（调整尺寸至模型输入）
            results = self.model.predict(
                source=crop,
                imgsz=self.window_size,
                conf=0.34,
                device="cpu",
                verbose=False
            )

            # 保存窗口预测结果
            window_pred_path = os.path.join(output_dir, f"{base_name}_window{idx}.jpg")
            cv2.imwrite(window_pred_path, results[0].plot())

            # 记录检测数据（坐标转换为全局）
            for box in results[0].boxes:
                x1_box, y1_box, x2_box, y2_box = box.xyxy[0].tolist()
                global_coords = (
                    x1 + x1_box, y1 + y1_box,
                    x1 + x2_box, y1 + y2_box
                )
                self.results_df = pd.concat([self.results_df, pd.DataFrame([{
                    "x1": global_coords[0],
                    "y1": global_coords[1],
                    "x2": global_coords[2],
                    "y2": global_coords[3],
                    "conf": box.conf.item(),
                    "cls": self.model.names[int(box.cls)],
                    "window_pos": f"({x1},{y1})-({x2},{y2})"
                }])], ignore_index=True)

            # 拼接预测结果到全图
            pred_plot = results[0].plot()
            merged_img[y1:y2, x1:x2] = cv2.addWeighted(
                merged_img[y1:y2, x1:x2], 0.3,
                pred_plot, 0.7, 0
            )
            mask[y1:y2, x1:x2] = 1

        # 保存最终结果
        cv2.imwrite(os.path.join(output_dir, f"{base_name}_merged.jpg"), merged_img)
        self.results_df.to_csv(os.path.join(output_dir, f"{base_name}_predictions.csv"), index=False)

        # 返回原始图像与预测叠加图
        return merged_img

# 使用示例
if __name__ == "__main__":
    predictor = SlidingWindowPredictor(
        model_path="/home/Mos/Documents/Complex/MyStudy/new_yolo/yolo-poSeg/runs/train3/weights/best.pt",
        window_size=1280,
        overlap=0.2
    )
    predictor.predict_and_save(
        image_path="/home/Mos/Desktop/mtemp/complete_test/Experimental_plot_01r.png",
        output_dir="/home/Mos/Desktop/mtemp/complete_test/pred5"
    )
