import pickle

# 指定 label.cache 文件的路径
cache_file_path = '/home/Mos/Documents/Complex/MyStudy/new_yolo/datasets/coco128-poseg/labels/train.cache'  # 请替换为实际文件路径

with open(cache_file_path, "rb") as f:
    data = pickle.load(f)

# 打印缓存内容
print("Cache keys:", data.keys())  # 通常包含'im_files', 'labels', 'n'等键
print("Number of samples:", data["n"])
print("Example labels:", data["labels"][0])  # 查看第一个样本的标签
