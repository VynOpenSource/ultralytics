# Ultralytics YOLO 🚀, AGPL-3.0 license

from copy import copy

import torch

from ultralytics.nn.tasks import RTDETRDetectionModel
from ultralytics.yolo.utils import DEFAULT_CFG, RANK, colorstr
from ultralytics.yolo.v8.detect import DetectionTrainer

from .val import RTDETRDataset, RTDETRValidator


class RTDETRTrainer(DetectionTrainer):

    def get_model(self, cfg=None, weights=None, verbose=True):
        """Return a YOLO detection model."""
        model = RTDETRDetectionModel(cfg, nc=self.data['nc'], verbose=verbose and RANK == -1)
        if weights:
            model.load(weights)
        return model

    def build_dataset(self, img_path, mode='val', batch=None):
        """Build RTDETR Dataset

        Args:
            img_path (str): Path to the folder containing images.
            mode (str): `train` mode or `val` mode, users are able to customize different augmentations for each mode.
            batch (int, optional): Size of batches, this is for `rect`. Defaults to None.
        """
        custom_args = getattr(self.args, 'cfg', {}) or {}
        custom_methods = self.methods or {}
        return RTDETRDataset(
            img_path=img_path,
            imgsz=self.args.imgsz,
            batch_size=batch,
            augment=mode == 'train',  # no augmentation
            hyp=self.args,
            rect=False,  # no rect
            cache=self.args.cache or None,
            prefix=colorstr(f'{mode}: '),
            # ISaac's addition
            ambiguous_classes=getattr(self.args, 'ambiguous_classes', None) or custom_args.get('ambiguous_classes', None),
            class_rate=getattr(self.args, 'class_rate', None) or custom_args.get('class_rate', None),
            mapping_classes=getattr(self.args, 'mapping_classes', None) or custom_args.get('mapping_classes', None),
            shuffle_class=getattr(self.args, 'shuffle_class', None) or custom_args.get('shuffle_class', None),
            data_names=self.data.get('names'),
            custom_augmentation=custom_methods.get('custom_augmentation', None),
            # ---------------------
            data=self.data)

    def get_validator(self):
        """Returns a DetectionValidator for RTDETR model validation."""
        self.loss_names = 'giou_loss', 'cls_loss', 'l1_loss'
        return RTDETRValidator(self.test_loader, save_dir=self.save_dir, args=copy(self.args))

    def preprocess_batch(self, batch):
        """Preprocesses a batch of images by scaling and converting to float."""
        batch = super().preprocess_batch(batch)
        bs = len(batch['img'])
        batch_idx = batch['batch_idx']
        gt_bbox, gt_class = [], []
        for i in range(bs):
            gt_bbox.append(batch['bboxes'][batch_idx == i].to(batch_idx.device))
            gt_class.append(batch['cls'][batch_idx == i].to(device=batch_idx.device, dtype=torch.long))
        return batch


def train(cfg=DEFAULT_CFG, use_python=False):
    """Train and optimize RTDETR model given training data and device."""
    model = 'rtdetr-l.yaml'
    data = cfg.data or 'coco128.yaml'  # or yolo.ClassificationDataset("mnist")
    device = cfg.device if cfg.device is not None else ''

    # NOTE: F.grid_sample which is in rt-detr does not support deterministic=True
    # NOTE: amp training causes nan outputs and end with error while doing bipartite graph matching
    args = dict(model=model,
                data=data,
                device=device,
                imgsz=640,
                exist_ok=True,
                batch=4,
                deterministic=False,
                amp=False)
    trainer = RTDETRTrainer(overrides=args)
    trainer.train()


if __name__ == '__main__':
    train()
