import numpy as np

def parse_label(line):
    """
    解析自定义数据集标签格式，返回关键点和分割点坐标
    Args:
        line: 包含标签数据的列表（字符串格式）
    Returns:
        keypoints: 关键点列表，每个元素为(px, py, visibility)元组
        segments: 分割点列表，每个元素为(x, y)坐标元组
    """
    # 转换为浮点数列表
    label = [float(x) for x in line]
    class_index = int(label[0])
    keypoints = []
    segments = []

    # 判断数据格式类型
    if class_index == 0:
        if len(label) > 4 and label[4] > 1:  # 格式二（包含bbox）
            header_len = 5  # class(1) + bbox(4)
        else:  # 格式一（无bbox）
            header_len = 1  # 仅class索引

        # 提取关键点部分
        remaining = label[header_len:]
        keypoint_data = []
        segment_data = []

        # 动态确定关键点和分割点分界
        max_keypoints = len(remaining) // 3
        for n in range(max_keypoints, -1, -1):
            seg_start = n * 3
            if seg_start <= len(remaining) and (len(remaining) - seg_start) % 2 == 0:
                keypoint_data = remaining[:seg_start]
                segment_data = remaining[seg_start:]
                break

        # 解析关键点（三元组）
        for i in range(0, len(keypoint_data), 3):
            px = keypoint_data[i]
            py = keypoint_data[i+1]
            visibility = int(keypoint_data[i+2])
            keypoints.append((px, py, visibility))

        # 解析分割点（二元组）
        for i in range(0, len(segment_data), 2):
            x = segment_data[i]
            y = segment_data[i+1]
            segments.append((x, y))
    else:
        # 非0类别直接解析分割点
        segment_data = label[1:]
        for i in range(0, len(segment_data), 2):
            if i+1 >= len(segment_data):
                break  # 处理奇数情况
            x = segment_data[i]
            y = segment_data[i+1]
            segments.append((x, y))

    return keypoints, segments

# 示例用法
line = ['38', '0.0992344', '0.644354', '0.101437', '0.665875', '0.0860313', '0.702063', '0.0735469', '0.748042', '0.0772188', '0.789125', '0.0940938', '0.808688', '0.117578', '0.813562', '0.136641', '0.799875', '0.147656', '0.773458', '0.146922', '0.745104', '0.139578', '0.719667', '0.130781', '0.701083', '0.113906', '0.672708', '0.108766', '0.631646']

# line = ['0', '0.589063', '0.257611', '2.000000', '0.595313', '0.250585', '2.000000', '0.582812', '0.248244', '2.000000']
label = np.array(line, dtype=np.float32)
keypoints, segments = parse_label(line)

print("关键点：", keypoints)
print("分割点：", segments)
print(len(keypoints), len(segments))

poly = np.array(segments)
print(poly)

assert (poly <= 1.0).all(), f"多边形坐标越界 {poly}"

repoly = poly.reshape(-1)
print(repoly)

print(len(repoly))

print('--------------------------------')
print(f'keypoints: {keypoints}')

# 遍历每个元组，检查前两位是否都小于1
for item in keypoints:
    if item[0] < 1 and item[1] < 1:
        print(f"元组 {item} 前两位都小于1")
    else:
        print(f"元组 {item} 前两位不满足条件")

for item in keypoints:
    if item[2] in {0.,1.,2.}:
        print(f"可见性标签合格")
    else:
        print(f"可见性标签不合格")
