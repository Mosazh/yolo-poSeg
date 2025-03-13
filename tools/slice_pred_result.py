'''

slice_x	slice_y	x_min	y_min	x_max	y_max	confidence	class_id	keypoints
0	    0	    100.0	150.0	200.0	250.0	0.95	    1	        (110.5, 160.3);...
'''
import os
import cv2
import numpy as np
from ultralytics import YOLO
import csv

# ========== 配置部分 ==========
MODEL_PATH = '/home/Mos/Downloads/train&val/20250111-20250112/train4/weights/best.pt'
# MODEL_PATH = '/home/Mos/Downloads/train&val/20250220-x/train1/weights/best.pt'

IMG_PATH = '/home/Mos/OtherDisks/TiPlus7100/jyx/jyx_mb_rgb.png'

# 切换到脚本所在目录
os.chdir(os.path.dirname(__file__))

# 输出结果目录
OUTPUT_DIR = '../output_pose/test7'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 滑动裁剪参数
SLICE_HEIGHT = 900
SLICE_WIDTH = 900
OVERLAP = 0.2  # 重叠比例
STEP_HEIGHT = int(SLICE_HEIGHT * (1 - OVERLAP))
STEP_WIDTH = int(SLICE_WIDTH * (1 - OVERLAP))

# ========== 初始化模型 ==========
model = YOLO(MODEL_PATH)

# ========== 读取图像 ==========
image = cv2.imread(IMG_PATH)
if image is None:
    print(f"无法读取图像: {IMG_PATH}")
    exit()

img_height, img_width = image.shape[:2]

# 创建结果图像的副本
result_image = image.copy()

# 检测结果容器
all_detections = []

# ========== 定义后处理函数 ==========
def nms(detections, iou_threshold=0.5):
    if len(detections) == 0:
        return []

    detections = np.array(detections)
    x_min, y_min, x_max, y_max, confidence = detections[:, :5].T

    # 转换数据类型
    confidence = confidence.astype(float)
    x_min = x_min.astype(float)
    y_min = y_min.astype(float)
    x_max = x_max.astype(float)
    y_max = y_max.astype(float)

    indices = np.argsort(-confidence)

    selected_indices = []
    while len(indices) > 0:
        current = indices[0]
        selected_indices.append(current)
        rest = indices[1:]

        xx1 = np.maximum(x_min[current], x_min[rest])
        yy1 = np.maximum(y_min[current], y_min[rest])
        xx2 = np.minimum(x_max[current], x_max[rest])
        yy2 = np.minimum(y_max[current], y_max[rest])

        inter_area = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        union_area = (
            (x_max[current] - x_min[current]) * (y_max[current] - y_min[current]) +
            (x_max[rest] - x_min[rest]) * (y_max[rest] - y_min[rest]) - inter_area
        )
        iou = inter_area / union_area

        indices = rest[iou < iou_threshold]

    return detections[selected_indices].tolist()

# CSV 文件路径
csv_file_path = os.path.join(OUTPUT_DIR, 'detection_results.csv')

# 打开 CSV 文件
with open(csv_file_path, mode='w', newline='') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(["slice_x", "slice_y", "x_min", "y_min", "x_max", "y_max", "confidence", "class_id", "keypoints"])

    # ========== 滑动裁剪并预测 ==========
    for y in range(0, img_height, STEP_HEIGHT):
        for x in range(0, img_width, STEP_WIDTH):
            # 裁剪图像
            slice_img = image[y:y + SLICE_HEIGHT, x:x + SLICE_WIDTH]
            if slice_img.shape[0] != SLICE_HEIGHT or slice_img.shape[1] != SLICE_WIDTH:
                pad_bottom = SLICE_HEIGHT - slice_img.shape[0]
                pad_right = SLICE_WIDTH - slice_img.shape[1]
                slice_img = cv2.copyMakeBorder(slice_img, 0, pad_bottom, 0, pad_right, cv2.BORDER_REPLICATE)

            # 推理
            results = model(slice_img)

            # 关键点数据
            keypoints = results[0].keypoints.data.cpu().numpy() if hasattr(results[0], 'keypoints') and results[0].keypoints is not None else []

            # ========== 映射检测框和关键点到全图坐标 ==========
            for i, box in enumerate(results[0].boxes):
                xyxy = box.xyxy[0].cpu().numpy()
                x_min, y_min, x_max, y_max = xyxy
                confidence = float(box.conf[0].item())
                class_id = int(box.cls[0].item())

                x_min += x
                y_min += y
                x_max += x
                y_max += y

                # 获取与框对应的关键点
                keypoints_str = ""
                if len(keypoints) > 0:
                    keypoint = keypoints[i]
                    keypoints_str = ';'.join([f"({kp[0]:.2f}, {kp[1]:.2f})" for kp in keypoint])

                all_detections.append([x, y, x_min, y_min, x_max, y_max, confidence, class_id, keypoints_str])

    # ========== 后处理：NMS 去除冗余框 ==========
    filtered_detections = nms(all_detections)

    # 保存最终检测结果到 CSV 文件并绘制到图像上
    for det in filtered_detections:
        slice_x, slice_y, x_min, y_min, x_max, y_max, confidence, class_id, keypoints_str = det
        writer.writerow(det)

        x_min_int = int(float(x_min))  # 转换为整数
        y_min_int = int(float(y_min))
        x_max_int = int(float(x_max))
        y_max_int = int(float(y_max))
        cv2.rectangle(result_image, (x_min_int, y_min_int), (x_max_int, y_max_int), (0, 255, 0), 2)

        # 绘制关键点
        # if keypoints_str:
        #     keypoints = [eval(kp) for kp in keypoints_str.split(';')]
        #     for kp in keypoints:
        #         cv2.circle(result_image, (int(kp[0]), int(kp[1])), 3, (0, 0, 255), -1)

        # 绘制关键点
        if keypoints_str:
            # 确保 slice_x 和 slice_y 是数值类型
            slice_x = float(slice_x)
            slice_y = float(slice_y)

            # 解析关键点字符串并转换为全图坐标
            keypoints = [eval(kp) for kp in keypoints_str.split(';')]
            for kp in keypoints:
                global_x = int(kp[0] + slice_x)  # 将局部坐标映射到全图坐标
                global_y = int(kp[1] + slice_y)
                cv2.circle(result_image, (global_x, global_y), 3, (0, 0, 255), -1)

    print(f"所有检测结果保存到: {csv_file_path}")

# ========== 保存拼接后的结果图像 ==========
final_output_path = os.path.join(OUTPUT_DIR, 'final_result.jpg')
cv2.imwrite(final_output_path, result_image)
print(f"拼接结果保存到: {final_output_path}")
