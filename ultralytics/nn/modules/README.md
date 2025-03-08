# PoSeg Modify Brochure

## PoSeg Head

**file path** : `yolo-poSeg/ultralytics/nn/modules/head.py`

``` python
""" custom MultiTask head """
class PoSeg(nn.Module):
    def __init__(self, nc=80, nm=32, npr=256, kpt_shape=(17, 3), ch=()):
        """Initialize the YOLO model attributes such as the number of masks, prototypes, and the convolution layers."""

        self.segment_head = Segment(nc, nm, npr, ch)
        self.pose_head = Pose(nc, kpt_shape, ch)
```

## model Yaml

**file path** : `yolo-poSeg/ultralytics/cfg/v8/yolov8-poseg.yaml`

``` yaml
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

# Ultralytics YOLOv8-pose keypoints/pose estimation model with P3/8 - P5/32 outputs
# Model docs: https://docs.ultralytics.com/models/yolov8
# Task docs: https://docs.ultralytics.com/tasks/pose

# # Thank https://github.com/ultralytics/ultralytics/issues/6949

# Parameters
nc: 1 # number of classes
kpt_shape: [17, 3] # number of keypoints, number of dims (2 for x,y or 3 for x,y,visible)
scales: # model compound scaling constants, i.e. 'model=yolov8n-pose.yaml' will call yolov8-pose.yaml with scale 'n'
  # [depth, width, max_channels]
  n: [0.33, 0.25, 1024]
  s: [0.33, 0.50, 1024]
  m: [0.67, 0.75, 768]
  l: [1.00, 1.00, 512]
  x: [1.00, 1.25, 512]

# YOLOv8.0n backbone
backbone:
  # [from, repeats, module, args]
  - [-1, 1, Conv, [64, 3, 2]] # 0-P1/2
  - [-1, 1, Conv, [128, 3, 2]] # 1-P2/4
  - [-1, 3, C2f, [128, True]]
  - [-1, 1, Conv, [256, 3, 2]] # 3-P3/8
  - [-1, 6, C2f, [256, True]]
  - [-1, 1, Conv, [512, 3, 2]] # 5-P4/16
  - [-1, 6, C2f, [512, True]]
  - [-1, 1, Conv, [1024, 3, 2]] # 7-P5/32
  - [-1, 3, C2f, [1024, True]]
  - [-1, 1, SPPF, [1024, 5]] # 9

# YOLOv8.0n head
head:
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 6], 1, Concat, [1]] # cat backbone P4
  - [-1, 3, C2f, [512]] # 12

  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 4], 1, Concat, [1]] # cat backbone P3
  - [-1, 3, C2f, [256]] # 15 (P3/8-small)

  - [-1, 1, Conv, [256, 3, 2]]
  - [[-1, 12], 1, Concat, [1]] # cat head P4
  - [-1, 3, C2f, [512]] # 18 (P4/16-medium)

  - [-1, 1, Conv, [512, 3, 2]]
  - [[-1, 9], 1, Concat, [1]] # cat head P5
  - [-1, 3, C2f, [1024]] # 21 (P5/32-large)

  # - [[15, 18, 21], 1, Pose, [nc, kpt_shape]] # Pose(P3, P4, P5)\
  # - [[15, 18, 21], 1, Segment, [nc, 32, 256]] # Segment(P3, P4, P5)
  - [[15, 18, 21], 1, PoSeg, [nc, 32, 256, kpt_shape]]

```

## PoSegModel

**file path** : `yolo-poSeg/ultralytics/nn/tasks.py`

``` python
""" custom task Model """
class PoSegModel(DetectionModel):
    """YOLO pose and segmenttation multitask model."""
    def __init__(self, cfg="yolov8-poseg.yaml", ch=3, nc=None, data_kpt_shape=(None, None), verbose=True):
        """Initialize YOLO pose and segmenttation multitask model with given config and parameters."""
        if not isinstance(cfg, dict):
            cfg = yaml_model_load(cfg)  # load model YAML
        if any(data_kpt_shape) and list(data_kpt_shape) != list(cfg["kpt_shape"]):
            LOGGER.info(f"Overriding model.yaml kpt_shape={cfg['kpt_shape']} with kpt_shape={data_kpt_shape}")
            cfg["kpt_shape"] = data_kpt_shape
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)

    def init_criterion(self):
        """Initialize the loss criterion for the SegmentationModel."""
        return v8PoSegLoss(self)
```

## PoSegLoss

**file path** : `yolo-poSeg/ultralytics/nn/loss.py`

_NOTIC:_
| Calculate the loss for instance segmentation\. |        |                                                                                           |          |
|------------------------------------------------|--------|-------------------------------------------------------------------------------------------|----------|
| Parameters:                                    |        |                                                                                           |          |
| Name                                           | Type   | Description                                                                               | Default  |
| fg\_mask                                       | Tensor | A binary tensor of shape \(BS, N\_anchors\) indicating which anchors are positive\.       | required |
| masks                                          | Tensor | Ground truth masks of shape \(BS, H, W\) if overlap is False, otherwise \(BS, ?, H, W\)\. | required |
| target\_gt\_idx                                | Tensor | Indexes of ground truth objects for each anchor of shape \(BS, N\_anchors\)\.             | required |
| target\_bboxes                                 | Tensor | Ground truth bounding boxes for each anchor of shape \(BS, N\_anchors, 4\)\.              | required |
| batch\_idx                                     | Tensor | Batch indices of shape \(N\_labels\_in\_batch, 1\)\.                                      | required |
| proto                                          | Tensor | Prototype masks of shape \(BS, 32, H, W\)\.                                               | required |
| pred\_masks                                    | Tensor | Predicted masks for each anchor of shape \(BS, N\_anchors, 32\)\.                         | required |
| imgsz                                          | Tensor | Size of the input image as a tensor of shape \(2\), i\.e\., \(H, W\)\.                    | required |
| overlap                                        | bool   | Whether the masks in masks tensor overlap\.                                               | required |
___

| Calculate the keypoints loss for the model\.                                                                                                                                                                                                                                                                          |        |                                                                                         |          |
|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------|-----------------------------------------------------------------------------------------|----------|
| This function calculates the keypoints loss and keypoints object loss for a given batch\. The keypoints loss is based on the difference between the predicted keypoints and ground truth keypoints\. The keypoints object loss is a binary classification loss that classifies whether a keypoint is present or not\. |        |                                                                                         |          |
| Parameters:                                                                                                                                                                                                                                                                                                           |        |                                                                                         |          |
| Name                                                                                                                                                                                                                                                                                                                  | Type   | Description                                                                             | Default  |
| masks                                                                                                                                                                                                                                                                                                                 | Tensor | Binary mask tensor indicating object presence, shape \(BS, N\_anchors\)\.               | required |
| target\_gt\_idx                                                                                                                                                                                                                                                                                                       | Tensor | Index tensor mapping anchors to ground truth objects, shape \(BS, N\_anchors\)\.        | required |
| keypoints                                                                                                                                                                                                                                                                                                             | Tensor | Ground truth keypoints, shape \(N\_kpts\_in\_batch, N\_kpts\_per\_object, kpts\_dim\)\. | required |
| batch\_idx                                                                                                                                                                                                                                                                                                            | Tensor | Batch index tensor for keypoints, shape \(N\_kpts\_in\_batch, 1\)\.                     | required |
| stride\_tensor                                                                                                                                                                                                                                                                                                        | Tensor | Stride tensor for anchors, shape \(N\_anchors, 1\)\.                                    | required |
| target\_bboxes                                                                                                                                                                                                                                                                                                        | Tensor | Ground truth boxes in \(x1, y1, x2, y2\) format, shape \(BS, N\_anchors, 4\)\.          | required |
| pred\_kpts                                                                                                                                                                                                                                                                                                            | Tensor | Predicted keypoints, shape \(BS, N\_anchors, N\_kpts\_per\_object, kpts\_dim\)\.        | required |

