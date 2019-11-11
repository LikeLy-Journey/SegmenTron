"""Base Model for Semantic Segmentation"""
import torch.nn as nn

from .backbones.resnet import get_resnet
from .backbones.xception import get_xception
from .backbones.mobilenet import get_mobilenet
from .backbones.hrnet import get_hrnet
from ..modules import get_norm
from ..config import cfg
__all__ = ['SegBaseModel']


class SegBaseModel(nn.Module):
    r"""Base Model for Semantic Segmentation
    """
    def __init__(self, nclass):
        super(SegBaseModel, self).__init__()
        self.nclass = nclass
        self.aux = cfg.SOLVER.AUX
        self.norm_layer = get_norm(cfg.MODEL.BN_TYPE)
        self.get_backbone()

    def get_backbone(self):
        self.backbone = cfg.MODEL.BACKBONE.lower()
        if self.backbone.startswith('xception'):
            self.encoder = get_xception(self.backbone, self.norm_layer)
        elif self.backbone.startswith('resnet'):
            self.encoder = get_resnet(self.backbone, self.norm_layer)
        elif self.backbone.startswith('mobilenet'):
            self.encoder = get_mobilenet(self.backbone, self.norm_layer)
        elif self.backbone.startswith('hrnet'):
            self.encoder = get_hrnet(self.backbone, self.norm_layer)
        else:
            raise RuntimeError('unknown backbone: {}'.format(self.backbone))

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
