import json
import os
import cv2

def yolo_pose_to_coco(
    label_dir: str,
    image_dir: str,
    output_json: str,
    num_keypoints: int = 17,
    keypoint_names: list = None,
    categories: list = None,
    visibility_map: dict = {0: 0, 1: 2}  # YOLO可见性到COCO的映射
):
    """
    将YOLO-Pose格式转换为COCO-Pose格式

    参数：
    label_dir: YOLO标签目录路径
    image_dir: 对应图片目录路径
    output_json: 输出JSON文件路径
    num_keypoints: 关键点数量（默认17）
    keypoint_names: 关键点名称列表（默认COCO格式）
    categories: 类别列表（默认单人格式）
    visibility_map: 可见性映射字典（YOLO到COCO）
    """
    # 初始化COCO数据结构
    coco = {
        "images": [],
        "annotations": [],
        "categories": []
    }

    # 配置默认关键点名称（COCO格式）
    if not keypoint_names:
        keypoint_names = [
            "nose", "left_eye", "right_eye", "left_ear", "right_ear",
            "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
            "left_wrist", "right_wrist", "left_hip", "right_hip",
            "left_knee", "right_knee", "left_ankle", "right_ankle"
        ]

    # 配置默认类别（单人检测）
    if not categories:
        categories = [{
            "id": 0,
            "name": "person",
            "supercategory": "person",
            "keypoints": keypoint_names,
            "skeleton": [
                [16,14],[14,12],[17,15],[15,13],[12,13],[6,12],[7,13],[6,7],
                [6,8],[7,9],[8,10],[9,11],[2,3],[1,2],[1,3],[2,4],[3,5],[4,6],[5,7]
            ]
        }]

    coco["categories"] = categories

    annotation_id = 1
    image_id = 1

    # 遍历所有标签文件
    for label_file in os.listdir(label_dir):
        if not label_file.endswith(".txt"):
            continue

        # 获取对应图片路径
        base_name = os.path.splitext(label_file)[0]
        image_path = os.path.join(image_dir, f"{base_name}.jpg")

        # 跳过不存在的图片
        if not os.path.exists(image_path):
            continue

        # 读取图片尺寸
        img = cv2.imread(image_path)
        height, width = img.shape[:2]

        # 添加图片信息
        coco["images"].append({
            "id": image_id,
            "file_name": os.path.basename(image_path),
            "width": width,
            "height": height
        })

        # 读取标签文件内容
        with open(os.path.join(label_dir, label_file)) as f:
            lines = [l.strip() for l in f.readlines()]

        for line in lines:
            if not line:
                continue

            parts = list(map(float, line.split()))

            # 验证数据完整性
            if len(parts) < 5 + num_keypoints * 3:
                print(f"Invalid line in {label_file}: {line}")
                continue

            # 解析基础信息
            category_id = int(parts[0])
            x_center = parts[1] * width
            y_center = parts[2] * height
            w = parts[3] * width
            h = parts[4] * height

            # 转换bbox格式
            bbox = [
                x_center - w/2,  # x
                y_center - h/2,  # y
                w,              # width
                h               # height
            ]

            # 处理关键点
            keypoints = []
            visible_count = 0

            for i in range(num_keypoints):
                idx = 5 + i*3
                try:
                    x = parts[idx] * width
                    y = parts[idx+1] * height
                    v = int(parts[idx+2])
                except IndexError:
                    x = y = 0.0
                    v = 0

                # 转换可见性
                coco_v = visibility_map.get(v, 0)
                if coco_v > 0:
                    visible_count += 1

                keypoints += [x, y, coco_v]

            # 构建annotation对象
            coco["annotations"].append({
                "id": annotation_id,
                "image_id": image_id,
                "category_id": category_id,
                "bbox": [round(x, 2) for x in bbox],
                "area": round(w * h, 2),
                "iscrowd": 0,
                "keypoints": [round(k, 2) if i%3 !=2 else int(k) for i,k in enumerate(keypoints)],
                "num_keypoints": visible_count
            })

            annotation_id += 1

        image_id += 1

    # 保存结果
    with open(output_json, 'w') as f:
        json.dump(coco, f, indent=2)

if __name__ == "__main__":
    # 使用示例
    yolo_pose_to_coco(
        label_dir="path/to/yolo/labels",
        image_dir="path/to/images",
        output_json="coco_pose.json",
        num_keypoints=17
    )
