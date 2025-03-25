import os
import sys
from datetime import datetime

def compute_iou(bbox1, bbox2):
    """计算两个边界框的交并比（IoU）"""
    x_min1, y_min1, x_max1, y_max1 = bbox1
    x_min2, y_min2, x_max2, y_max2 = bbox2

    inter_x_min = max(x_min1, x_min2)
    inter_y_min = max(y_min1, y_min2)
    inter_x_max = min(x_max1, x_max2)
    inter_y_max = min(y_max1, y_max2)

    inter_area = max(0, inter_x_max - inter_x_min) * max(0, inter_y_max - inter_y_min)
    area1 = (x_max1 - x_min1) * (y_max1 - y_min1)
    area2 = (x_max2 - x_min2) * (y_max2 - y_min2)
    union_area = area1 + area2 - inter_area

    return inter_area / union_area if union_area > 0 else 0.0

def parse_seg_line(seg_parts):
    """解析seg标签行并计算边界框"""
    coords = list(map(float, seg_parts[1:]))
    x_coords = coords[::2]
    y_coords = coords[1::2]
    return (min(x_coords), min(y_coords), max(x_coords), max(y_coords))

def parse_pose_line(pose_parts):
    """解析pose标签行并提取关键点信息"""
    x, y, w, h = map(float, pose_parts[1:5])
    bbox = (x - w/2, y - h/2, x + w/2, y + h/2)
    return bbox, pose_parts[1:5], pose_parts[5:]

def process_single_pair(seg_lines, pose_lines):
    """处理单个文件对的合并逻辑"""
    # 预处理pose数据
    pose_data = []
    for pl in pose_lines:
        parts = pl.split()
        if len(parts) >= 5:
            try:
                bbox, Obbox, kpts = parse_pose_line(parts)
                pose_data.append((bbox, Obbox, kpts))
            except:
                continue

    merged_lines = []
    for seg_line in seg_lines:
        seg_parts = seg_line.split()
        if not seg_parts:
            continue

        # 非类别0直接保留
        if seg_parts[0] != '0':
            merged_lines.append(seg_line)
            continue

        try:
            seg_bbox = parse_seg_line(seg_parts)
            best_kpts = []
            best_bbox = []
            best_iou = 0.0

            # 寻找最佳匹配pose
            for pose_bbox, Obbox, kpts in pose_data:
                current_iou = compute_iou(seg_bbox, tuple(map(float, pose_bbox)))
                if current_iou > best_iou:
                    best_iou = current_iou
                    best_kpts = kpts
                    best_bbox = Obbox

            # 构造新行
            if best_kpts:
                new_line = [seg_parts[0]] + list(best_bbox) + best_kpts + seg_parts[1:]
                merged_lines.append(' '.join(map(str, new_line)))
            else:
                merged_lines.append(seg_line)

        except Exception as e:
            print(f"处理行时出错: {e}")
            merged_lines.append(seg_line)  # 保留原始行

    return merged_lines

def batch_merge_labels(pose_dir, seg_dir, output_dir):
    """批量处理的主逻辑"""
    os.makedirs(output_dir, exist_ok=True)
    seg_files = {os.path.splitext(f)[0]: f for f in os.listdir(seg_dir) if f.endswith('.txt')}

    processed = 0
    error_files = []

    for pose_file in os.listdir(pose_dir):
        if not pose_file.endswith('.txt'):
            continue

        base_name = os.path.splitext(pose_file)[0]
        seg_file = seg_files.get(base_name)
        if not seg_file:
            continue

        try:
            # 读取文件内容
            with open(os.path.join(seg_dir, seg_file), 'r') as f:
                seg_content = [line.strip() for line in f if line.strip()]
            with open(os.path.join(pose_dir, pose_file), 'r') as f:
                pose_content = [line.strip() for line in f if line.strip()]

            # 处理合并
            merged = process_single_pair(seg_content, pose_content)

            # 写入输出
            output_path = os.path.join(output_dir, pose_file)
            with open(output_path, 'w') as f:
                f.write('\n'.join(merged))

            processed += 1
            print(f"已处理: {pose_file} → {output_path}")

        except Exception as e:
            error_files.append((pose_file, str(e)))
            print(f"处理 {pose_file} 失败: {str(e)}", file=sys.stderr)

    print(f"\n处理完成！成功处理 {processed} 个文件，失败 {len(error_files)} 个")

if __name__ == '__main__':
    seg_labels_path = '/home/Mos/OtherDisks/TiPlus7100/Datasets/COCO/coco-seg_labels/labels/val2017'
    pose_labels_path = '/home/Mos/OtherDisks/TiPlus7100/Datasets/COCO/coco-pose_labels/labels/val2017'
    output_path= '/home/Mos/OtherDisks/TiPlus7100/Datasets/COCO/coco-poseg_labels(kpts_box)'

    batch_merge_labels(pose_labels_path, seg_labels_path, output_path)
