from ultralytics import YOLO

def main(model="yolov8-poseg.yaml",
         data="MKSD.yaml",
         epochs=2,
         imgsz=640,
         batch=16,
         multi_scale=True,
         degrees=180,
         box_lw=30,                        # loss weight of box is 30,
         scale=0.5,
         shear=0.5,
         flipud=1,
         fliplr=1,
         mixup=0.5,
         cutmix=0.5,
         iouType="Ciou",
         nwd_loss=False,
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
        multi_scale=multi_scale,
        degrees=degrees,
        box=box_lw,
        scale=scale,
        shear=shear,
        flipud=flipud,
        fliplr=fliplr,
        mixup=mixup,
        cutmix=cutmix,
        iou_type=iouType,
        nwd_loss=nwd_loss,
        Inner_iou=Inner_iou,
        Focal=Focal,
        Focaler=Focaler,
        alpha=loss_alpha,
    )


if __name__ == '__main__':
    MODEL = "yolov8-poseg_ELA.yaml"
    DATA = "MKSD.yaml"
    EPOCHS = 300
    IMG_SIZE = 640
    BATCH = 16
    MULTI_SCALE = True
    DEGREES = 180
    BOX_LW = 30         # loss weight of box is 30,
    SCALE = 0.5
    SHEAL = 0.5           # Shears the image by a specified degree, mimicking the effect of objects being viewed from different angles.
    FLIPUD = 1
    FLIPLR = 1
    CUTMIX = 0.5
    MIXUP = 0
    IOU_TYPE = "Siou" # Ciou, Giou, Diou, Siou, Eiou, Wise-iou, MPDiou, Shape-iou, Powerful-iou, Unified-iou
    NWD_LOSS = False
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
         multi_scale=MULTI_SCALE,
         degrees=DEGREES,
         box_lw=BOX_LW,
         scale=SCALE,
         shear=SHEAL,
         flipud=FLIPUD,
         fliplr=FLIPLR,
         mixup=MIXUP,
         cutmix=CUTMIX,
         iouType=IOU_TYPE,
         nwd_loss=NWD_LOSS,
         Inner_iou=INNER_IOU,
         Focal=FOCAL,
         Focaler=FOCALER,
         loss_alpha=LOSS_ALPHA,
        )
