# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
# https://github.com/stedavkle/ultralytics/blob/multitask/ultralytics/models/yolo/multitask/predict.py

from ultralytics.models.yolo.detect.predict import DetectionPredictor
from ultralytics.utils import DEFAULT_CFG, LOGGER, ops
from ultralytics.engine.results import Results

class PoSegPredictor(DetectionPredictor):
    """
    A class extending the DetectionPredictor class for prediction based on a PoSegTask model.

    Example:
        ```python
        from ultralytics.utils import ASSETS
        from ultralytics.models.yolo.poseg import PoSegTaskPredictor

        args = dict(model="yolov8-poseg.pt", source=ASSETS)
        predictor = PoSegTaskPredictor(overrides=args)
        predictor.predict_cli()
        ```
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """Initializes the PoSegTaskPredictor with the provided configuration, overrides, and callbacks."""
        super().__init__(cfg, overrides, _callbacks)
        self.args.task = "poseg"

    def postprocess(self, preds, img, orig_imgs):
        """
        Applies non-max suppression and processes detections for each image in an input batch.

        Predicted segmentation masks and keypoints get handles separately.
        """
        p_seg = ops.non_max_suppression(
            preds[0][0],
            self.args.conf,
            self.args.iou,
            agnostic=self.args.agnostic_nms,
            max_det=self.args.max_det,
            classes=self.args.classes,
            nc=len(self.model.names),
        )
        p_pose = ops.non_max_suppression(
            preds[0][1] if len(preds[1]) == 4 else preds[1],
            self.args.conf,
            self.args.iou,
            agnostic=self.args.agnostic_nms,
            max_det=self.args.max_det,
            classes=self.args.classes,
            nc=len(self.model.names),
        )

        if not isinstance(orig_imgs, list):  # input images are a torch.Tensor, not a list
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)

        results = []
        protos = (
            preds[1][-2] if len(preds[1]) == 4 else preds[0][1]
        )  # second output is len 4 if pt, but only 1 if exported
        for i, (pred_seg, pred_kpt) in enumerate(zip(p_seg, p_pose)):
            pred_seg.clone()
            pred_kpt_copy = pred_kpt.clone()
            orig_img = orig_imgs[i]
            img_path = self.batch[0][i]

            # keypoint extraction
            pred_kpt_copy[:, :4] = ops.scale_boxes(img.shape[2:], pred_kpt_copy[:, :4], orig_img.shape).round()
            pred_kpts = (
                pred_kpt[:, 6:].view(len(pred_kpt_copy), *self.model.kpt_shape)
                if len(pred_kpt_copy)
                else pred_kpt_copy[:, 6:]
            )
            pred_kpts = ops.scale_coords(img.shape[2:], pred_kpts, orig_img.shape)

            # segmentation extraction
            if not len(pred_seg):  # save empty boxes
                masks = None
            elif self.args.retina_masks:
                pred_seg[:, :4] = ops.scale_boxes(img.shape[2:], pred_seg[:, :4], orig_img.shape)
                masks = ops.process_mask_native(protos[i], pred_seg[:, 6:], pred_seg[:, :4], orig_img.shape[:2])  # HWC
            else:
                masks = ops.process_mask(
                    protos[i], pred_seg[:, 6:], pred_seg[:, :4], img.shape[2:], upsample=True
                )  # HWC
                pred_seg[:, :4] = ops.scale_boxes(img.shape[2:], pred_seg[:, :4], orig_img.shape)
            results.append(
                Results(
                    orig_img,
                    path=img_path,
                    names=self.model.names,
                    boxes=pred_seg[:, :6],
                    masks=masks,
                    keypoints=pred_kpts,
                )
            )
        return results
