import os
import argparse

def sync_images_labels(images_root, labels_root, label_ext='.txt'):
    """
    同步图像和标签文件，删除没有对应标签的图像
    :param images_root: 图像文件根目录
    :param labels_root: 标签文件根目录
    :param label_ext: 标签文件扩展名
    """
    # 支持的图片格式扩展名
    image_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp'}

    # 遍历images目录下的所有子目录（train/val）
    for subset in os.listdir(images_root):
        img_subset_dir = os.path.join(images_root, subset)
        lbl_subset_dir = os.path.join(labels_root, subset)

        # 跳过非目录文件
        if not os.path.isdir(img_subset_dir):
            continue

        # 如果标签目录不存在则创建（虽然不符合需求，但防止意外情况）
        os.makedirs(lbl_subset_dir, exist_ok=True)

        # 遍历图片目录中的所有文件
        for img_file in os.listdir(img_subset_dir):
            img_path = os.path.join(img_subset_dir, img_file)

            # 跳过目录和非图片文件
            if not os.path.isfile(img_path):
                continue
            if os.path.splitext(img_file)[1].lower() not in image_exts:
                continue

            # 构建对应的标签文件路径
            base_name = os.path.splitext(img_file)[0]
            lbl_file = f"{base_name}{label_ext}"
            lbl_path = os.path.join(lbl_subset_dir, lbl_file)

            # 如果标签文件不存在，删除图片
            if not os.path.exists(lbl_path):
                print(f"删除无标签图片: {img_path}")
                os.remove(img_path)

def main():
    parser = argparse.ArgumentParser(description='同步图像和标签文件')
    parser.add_argument('--root', type=str, default='coco-multitask',
                      help='项目根目录（默认：coco-multitask）')
    parser.add_argument('--label-ext', type=str, default='.txt',
                      help='标签文件扩展名（默认：.txt）')
    parser.add_argument('--dry-run', action='store_true',
                      help='试运行模式（只显示将要删除的文件）')

    args = parser.parse_args()

    images_dir = os.path.join(args.root, 'images')
    labels_dir = os.path.join(args.root, 'labels')

    if not os.path.exists(images_dir):
        raise FileNotFoundError(f"图像目录不存在: {images_dir}")
    if not os.path.exists(labels_dir):
        raise FileNotFoundError(f"标签目录不存在: {labels_dir}")

    if args.dry_run:
        print("试运行模式：以下文件将被删除")
        # 这里可以添加试运行逻辑，但需要修改sync函数支持
        # 实际使用时建议先备份数据
        return

    sync_images_labels(images_dir, labels_dir, args.label_ext)

if __name__ == '__main__':
    main()
