# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
# https://github.com/stedavkle/ultralytics/blob/multitask/ultralytics/models/yolo/multitask/train.py

from copy import copy
from ultralytics.models import yolo
from ultralytics.nn.tasks import PoSegModel
from ultralytics.utils import DEFAULT_CFG, LOGGER, RANK
from ultralytics.utils.plotting import plot_images, plot_results
from ultralytics.models.yolo.detect import DetectionTrainer

class PoSegTrainer(DetectionTrainer):
    """
    A class extending the DetectionTrainer class for training based on a pose and segmentation model.

    Example:
        ```python
        from ultralytics.models.yolo.poSeg import PoSegTrainer

        args = dict(model="yolov8-poSeg.pt", data="coco8-poSeg.yaml", epochs=3)
        trainer = PoSegTrainer(overrides=args)
        trainer.train()
        ```
    """
    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """Initialize a PoseTrainer object with specified configurations and overrides."""
        """Initialize a SegmentationTrainer object with given arguments."""
        if overrides is None:
            overrides = {}
        overrides["task"] = "poseg"
        super().__init__(cfg, overrides, _callbacks)

        if isinstance(self.args.device, str) and self.args.device.lower() == "mps":
            LOGGER.warning(
                "WARNING ⚠️ Apple MPS known Pose bug. Recommend 'device=cpu' for Pose models. "
                "See https://github.com/ultralytics/ultralytics/issues/4031."
            )

    def get_model(self, cfg=None, weights=None, verbose=True):
        """Get PoSeg estimation model with specified configuration and weights."""
        model = PoSegModel(cfg, ch=3, nc=self.data["nc"], data_kpt_shape=self.data["kpt_shape"], verbose=verbose and RANK == -1)
        if weights:
            model.load(weights)

        return model

    def set_model_attributes(self):
        """Sets keypoints shape attribute of MultiTaskModel."""
        super().set_model_attributes()
        self.model.kpt_shape = self.data["kpt_shape"]

    def get_validator(self):
        """Returns an instance of the PoSegValidator class for validation."""
        self.loss_names = "box_loss", "seg_loss", "cls_loss", "dfl_loss", "pose_loss", "kobj_loss"
        return yolo.poseg.PoSegValidator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )

    def plot_training_samples(self, batch, ni):
        """Plot a batch of training samples with annotated class labels, bounding boxes, mask, and keypoints."""
        images = batch["img"]
        kpts = batch["keypoints"]
        masks=batch["masks"],
        cls = batch["cls"].squeeze(-1)
        bboxes = batch["bboxes"]
        paths = batch["im_file"]
        batch_idx = batch["batch_idx"]
        plot_images(
            images,
            batch_idx,
            cls,
            bboxes,
            masks=masks,
            kpts=kpts,
            paths=paths,
            fname=self.save_dir / f"train_batch{ni}.jpg",
            on_plot=self.on_plot,
        )
    def plot_metrics(self):
        """Plots training/val metrics."""
        plot_results(file=self.csv, poseg=True, on_plot=self.on_plot)  # save results.png
