import os
import cv2
import numpy as np
from ultralytics import YOLO
import csv
from tqdm import tqdm

# ========== 配置部分 ==========
# 切换到脚本所在目录
os.chdir(os.path.dirname(__file__))

MODEL_PATH = '/home/Mos/Documents/Complex/MyStudy/new_yolo/yolo-poSeg/runs/train3/weights/best.pt'
IMG_PATH = '/home/Mos/Desktop/mtemp/complete_test/Experimental_plot_01.png'
OUTPUT_DIR = '/home/Mos/Desktop/mtemp/complete_test/pred3'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 滑动窗口参数
SLICE_SIZE = 1200
OVERLAP = 0.2
STEP = int(SLICE_SIZE * (1 - OVERLAP))

# Mask参数
MASK_THRESHOLD = 0.5  # 二值化阈值[8](@ref)
MASK_ALPHA = 0.3      # 可视化透明度

# ========== 初始化模型 ==========
model = YOLO(MODEL_PATH)

# ========== 图像处理 ==========
image = cv2.imread(IMG_PATH)
h, w = image.shape[:2]
result_image = image.copy()

# 初始化全局mask画布[7](@ref)
global_mask = np.zeros((h, w), dtype=np.float32)

# 存储检测结果
all_detections = []

# ========== 滑动窗口处理 ==========
for y in tqdm(range(0, h, STEP), desc="Processing slices"):
    for x in range(0, w, STEP):
        # 计算当前切片坐标
        y1, y2 = y, min(y+SLICE_SIZE, h)
        x1, x2 = x, min(x+SLICE_SIZE, w)
        slice_img = image[y1:y2, x1:x2]

        # 填充不足尺寸的切片
        pad_h = SLICE_SIZE - (y2-y1)
        pad_w = SLICE_SIZE - (x2-x1)
        if pad_h > 0 or pad_w > 0:
            slice_img = cv2.copyMakeBorder(slice_img, 0, pad_h, 0, pad_w,
                                         cv2.BORDER_REPLICATE)

        # 模型推理[6](@ref)
        results = model(slice_img)

        # 处理每个检测结果
        for i, box in enumerate(results[0].boxes):
            # 解析边界框
            xyxy = box.xyxy[0].cpu().numpy()
            conf = box.conf[0].item()
            cls_id = int(box.cls[0].item())

            # 转换到全局坐标
            x_min = xyxy[0] + x
            y_min = xyxy[1] + y
            x_max = xyxy[2] + x
            y_max = xyxy[3] + y

            # 处理mask数据[8](@ref)
            if results[0].masks is not None:
                mask = results[0].masks[i].data[0].cpu().numpy()

                # 创建局部mask画布
                local_mask = np.zeros((SLICE_SIZE, SLICE_SIZE), dtype=np.float32)
                local_mask[:mask.shape[0], :mask.shape[1]] = mask

                # 映射到全局坐标系[7](@ref)
                global_y1 = y
                global_y2 = min(y+SLICE_SIZE, h)
                global_x1 = x
                global_x2 = min(x+SLICE_SIZE, w)

                # 更新全局mask
                global_mask[global_y1:global_y2, global_x1:global_x2] = np.maximum(
                    global_mask[global_y1:global_y2, global_x1:global_x2],
                    local_mask[:global_y2-global_y1, :global_x2-global_x1]
                )

            # 存储检测结果
            all_detections.append([
                x, y, x_min, y_min, x_max, y_max, conf, cls_id
            ])

# ========== Mask后处理 ==========
# 二值化处理[1](@ref)
binary_mask = (global_mask > MASK_THRESHOLD).astype(np.uint8) * 255

# 查找轮廓[1,8](@ref)
contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 过滤小面积区域
min_area = 100
filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]

# ========== 结果保存 ==========
with open(os.path.join(OUTPUT_DIR, 'results.csv'), 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["x_min", "y_min", "x_max", "y_max", "confidence", "class_id", "contour_points"])

    for det in all_detections:
        # 关联最近的轮廓
        x_center = (det[2] + det[4]) / 2
        y_center = (det[3] + det[5]) / 2
        nearest_contour = min(filtered_contours,
                            key=lambda c: cv2.pointPolygonTest(c, (x_center, y_center), True))

        # 转换轮廓点格式
        points = [[int(pt[0][0]), int(pt[0][1])] for pt in nearest_contour]

        writer.writerow([
            det[2], det[3], det[4], det[5], det[6], det[7],
            ';'.join(f"{x},{y}" for x,y in points)
        ])

# ========== 可视化 ==========
# 绘制mask叠加层[6](@ref)
mask_layer = np.zeros_like(image)
mask_layer[..., 1] = binary_mask  # 绿色通道
mask_layer = cv2.addWeighted(image, 1, mask_layer, MASK_ALPHA, 0)

# 绘制边界框
for det in all_detections:
    x_min, y_min, x_max, y_max = map(int, det[2:6])
    cv2.rectangle(mask_layer, (x_min, y_min), (x_max, y_max), (0,255,0), 2)

# 保存结果
cv2.imwrite(os.path.join(OUTPUT_DIR, 'result.jpg'), mask_layer)
print(f"Results saved to {OUTPUT_DIR}")
