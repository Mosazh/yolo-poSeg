import os
import shutil

os.chdir(os.path.dirname(__file__))
def extract_labels(source_dir, target_dir):
    # 创建目标文件夹
    os.makedirs(target_dir, exist_ok=True)

    # 遍历源文件夹中的所有txt文件
    for filename in os.listdir(source_dir):
        if not filename.endswith('.txt'):
            continue

        source_path = os.path.join(source_dir, filename)
        count_class0 = 0

        # 统计class-index=0的数量
        with open(source_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if not parts:  # 跳过空行
                    continue
                try:
                    class_idx = int(parts[0])
                    if class_idx == 0:
                        count_class0 += 1
                except ValueError:
                    continue  # 忽略格式错误的行

        # 如果数量为1，复制文件到目标目录
        if count_class0 == 1:
            target_path = os.path.join(target_dir, filename)
            shutil.copy2(source_path, target_path)
            print(f"已提取文件: {filename}")

if __name__ == "__main__":
    source_dir = '/home/Mos/OtherDisks/TiPlus7100/Datasets/COCO/coco-poseg_labels(kpts_box)/source_labels'
    target_dir = '/home/Mos/OtherDisks/TiPlus7100/Datasets/COCO/coco-poseg_labels(kpts_box)/new_labels'
    extract_labels(source_dir, target_dir)
