"""Base Model for Semantic Segmentation"""
import torch.nn as nn

from .backbones import get_segmentation_backbone
from ..data.dataloader import datasets
from ..modules import get_norm
from ..config import cfg
__all__ = ['SegBaseModel']


class SegBaseModel(nn.Module):
    r"""Base Model for Semantic Segmentation
    """
    def __init__(self, need_backbone=True):
        super(SegBaseModel, self).__init__()
        self.nclass = datasets[cfg.DATASET.NAME].NUM_CLASS
        self.aux = cfg.SOLVER.AUX
        self.norm_layer = get_norm(cfg.MODEL.BN_TYPE)
        self.backbone = None
        self.encoder = None
        if need_backbone:
            self.get_backbone()

    def get_backbone(self):
        self.backbone = cfg.MODEL.BACKBONE.lower()
        self.encoder = get_segmentation_backbone(self.backbone, self.norm_layer)

    def base_forward(self, x):
        """forwarding backbone network"""
        c1, c2, c3, c4 = self.encoder(x)
        return c1, c2, c3, c4

    def evaluate(self, x):
        """evaluating network with inputs and targets"""
        return self.forward(x)[0]

    def demo(self, x):
        pred = self.forward(x)
        if self.aux:
            pred = pred[0]
        return pred
