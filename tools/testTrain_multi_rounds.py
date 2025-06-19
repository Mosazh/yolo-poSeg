import os
import json
import time
import logging
import argparse
from ultralytics import YOLO

def main(params):
    model = YOLO(params.get('model', "yolov8-poseg.yaml"))
    results = model.train(
        data=params.get('data', "MKSD.yaml"),
        epochs=params.get('epochs', 1),
        imgsz=params.get('imgsz', 640),
        batch=params.get('batch', 16),
        multi_scale=params.get('multi_scale', False),
        degrees=params.get('degrees', 0),
        box=params.get('box_lw', 30),
        scale=params.get('scale', 0.5),
        shear=params.get('shear', 0),
        flipud=params.get('flipud', 0),
        fliplr=params.get('fliplr', 0),
        mixup=params.get('mixup', 0),
        cutmix=params.get('cutmix', 0),
        iou_type=params.get('iouType', 'iou'),
        Inner_iou=params.get('Inner_iou', False),
        Focal=params.get('Focal', False),
        Focaler=params.get('Focaler', False),
        alpha=params.get('loss_alpha', 1),
        name=params.get('name', 'train'),
    )

def parse_arguments():
    parser = argparse.ArgumentParser(description='Yolo-PoSeg Training Parameters')
    parser.add_argument('--model', default="yolov8-poseg.yaml", help='Model config file')
    parser.add_argument('--data', default="MKSD.yaml", help='Dataset config file')
    parser.add_argument('--epochs', type=int, default=1, help='Training epochs')
    parser.add_argument('--imgsz', type=int, default=640, help='Input image size')
    parser.add_argument('--batch', type=int, default=16, help='Batch size')
    parser.add_argument('--multi-scale', action='store_true', help='Multi-scale training')
    parser.add_argument('--degrees', type=int, default=0, help='Random rotation degrees')
    parser.add_argument('--box_lw', type=float, default=30, help='Box loss weight')
    parser.add_argument('--scale', type=float, default=0.5, help='Random scale range')
    parser.add_argument('--shear', type=float, default=0, help='Random shear range')  # 0.5
    parser.add_argument('--flipud', type=int, default=0, help='Random flip up-down')
    parser.add_argument('--fliplr', type=int, default=0, help='Random flip left-right')
    parser.add_argument('--mixup', type=float, default=0, help='Mixup alpha')
    parser.add_argument('--cutmix', type=float, default=0, help='Cutmix alpha')
    parser.add_argument('--iouType', default='Ciou', help='IoU type')
    parser.add_argument('--Inner_iou', action='store_true', help='Use Focal loss')
    parser.add_argument('--Focal', action='store_true', help='Use Focal loss')
    parser.add_argument('--Focaler', action='store_true', help='Use Focaler loss')
    parser.add_argument('--loss_alpha', type=float, default=0.25, help='Loss alpha')
    parser.add_argument('--name', default='train1', help='Model name')
    return vars(parser.parse_args())

if __name__ == '__main__':
    # 支持从命令行参数获取
    # params_from_cli = parse_arguments()
    # main(params_from_cli)

    # 配置日志记录，设置日志文件的完整路径
    log_file_path = "training_results.log"
    logging.basicConfig(filename=log_file_path, level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
    current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    logging.exception(f"Start training at {current_time}")
    # 直接在代码中传递参数
    try:
        with open("tools/train_params.json", "r") as f:
            pfj = json.load(f)                          # params_from_json
    except FileNotFoundError as e:
        raise FileNotFoundError("train_params.json文件不存在，请检查文件路径。")

    params_directly = {
        'model': "yolov8-poseg.yaml",
        'data': "MKSD.yaml",
        'epochs': 300,
        'imgsz': 640,
        'batch': 16,
        'multi_scale': True,
        'degrees': 180,
        'box_lw': 30,
        'scale': 0.5,
        'shear': 0.5,
        'flipud': 1,
        'fliplr': 1,
        'mixup': 0,
        'cutmix': 0.5,
        'iouType': "Ciou",          # Shears the image by a specified degree, mimicking the effect of objects being viewed from different angles.
        'Inner_iou': False,
        'Focal': False,
        'Focaler': False,
        'loss_alpha': 1,
    }

    for model in pfj["model"].values():
        for iouType in pfj["iouType"].values():
            try:
                # 构造参数
                params_directly["model"] = f"{model}.yaml"
                params_directly["iouType"] = iouType
                dirname = params_directly["model"].split('.')[0] + '_' + params_directly['iouType']
                params_directly["name"] = dirname

                if params_directly['batch'] > 32:
                    if "acmix" in params_directly["model"]:
                        params_directly["batch"] = 8
                    if "ELA" in params_directly["model"]:
                        params_directly["batch"] = 32
                    if "MobileNetV4" in params_directly["model"]:
                        params_directly["batch"] = 16

                # 调用主函数
                main(params_directly)

                # 记录成功的日志
                logging.info(json.dumps({
                    'model': params_directly["model"],
                    'iouType': params_directly["iouType"],
                    'status': 'success'
                }))

            except Exception as e:
                # 打印并记录错误信息
                error_message = f"Error occurred with model={model}, iouType={iouType}: {str(e)}"
                print(error_message)

                # 记录失败的日志
                logging.error(json.dumps({
                    'model': f"{model}.yaml",
                    'iouType': iouType,
                    'status': 'failed',
                    'error': str(e)
                }))
