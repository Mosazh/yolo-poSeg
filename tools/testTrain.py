from ultralytics import YOLO

def main(model="yolov8-poseg_ALSS.yaml",
         data="MKSD.yaml",
         epochs=2,
         imgsz=640,
         batch=16,
         degrees=180,
         box_lw=30,                        # loss weight of box is 30,
         iouType="Ciou",
         Inner_iou=False,
         Focal=False,
         Focaler=False,
         loss_alpha=1):
    # Load a model
    model = YOLO(model)

    # Train the model
    # results = model.train(data="cocoTest-poseg.yaml", device='cuda', optimizer='Adam', epochs=100, imgsz=640, batch=32)
    results = model.train(
        data=data,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        degrees=degrees,
        box=box_lw,
        iou_type=iouType,
        Inner_iou=Inner_iou,
        Focal=Focal,
        Focaler=Focaler,
        alpha=loss_alpha,
    )

# # Siou
# model.train(
    # imgsz=640,
#     epochs=1, workers=4, batch=8,
#             iou_type="Siou", Inner_iou=False, Focal=False, Focaler=False)

if __name__ == '__main__':
    MODEL = "yolov8-poseg_ConvNeXtv2.yaml"
    DATA = "MKSD.yaml"
    EPOCHS = 1
    IMG_SIZE = 640
    BATCH = 16
    DEGREES = 180
    BOX_LW = 30
    IOU_TYPE = "Siou" # Ciou, Giou, Diou, Siou, Eiou, Wise-iou, MPDiou, Shape-iou, Powerful-iou, Unified-iou
    INNER_IOU = False
    FOCAL = False
    FOCALER = False
    LOSS_ALPHA = 1 # AlphaIoU set to 3, IOU_TYPE set to 'iou'

    main(
         model=MODEL,
         data=DATA,
         epochs=EPOCHS,
         imgsz=IMG_SIZE,
         batch=BATCH,
         degrees=DEGREES,
         box_lw=BOX_LW,
         iouType=IOU_TYPE,
         Inner_iou=INNER_IOU,
         Focal=FOCAL,
         Focaler=FOCALER,
         loss_alpha=LOSS_ALPHA
        )
