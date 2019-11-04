"""Base Model for Semantic Segmentation"""
import torch.nn as nn

# from ..modules import JPU
from .backbones.resnet import get_resnet#resnet50_v1s, resnet101_v1s, resnet152_v1s
from .backbones.xception import get_xception
from ..utils.config import cfg
__all__ = ['SegBaseModel']


class SegBaseModel(nn.Module):
    r"""Base Model for Semantic Segmentation

    Parameters
    ----------
    backbone : string
        Pre-trained dilated backbone network type (default:'resnet50'; 'resnet50',
        'resnet101' or 'resnet152').
    """

    def __init__(self, nclass):
        super(SegBaseModel, self).__init__()
        self.aux = cfg.SOLVER.AUX
        self.nclass = nclass
        self.get_backbone()

    def get_backbone(self):
        backbone = cfg.MODEL.BACKBONE.lower()
        if backbone.startswith('xception'):
            self.encoder = get_xception(backbone)
        elif backbone.startswith('resnet'):
            self.encoder = get_resnet(backbone)
        else:
            raise RuntimeError('unknown backbone: {}'.format(backbone))

    def base_forward(self, x):
        """forwarding pre-trained network"""
        x = self.pretrained.conv1(x)
        x = self.pretrained.bn1(x)
        x = self.pretrained.relu(x)
        x = self.pretrained.maxpool(x)
        c1 = self.pretrained.layer1(x)
        c2 = self.pretrained.layer2(c1)
        c3 = self.pretrained.layer3(c2)
        c4 = self.pretrained.layer4(c3)

        return c1, c2, c3, c4

    def evaluate(self, x):
        """evaluating network with inputs and targets"""
        return self.forward(x)[0]

    def demo(self, x):
        pred = self.forward(x)
        if self.aux:
            pred = pred[0]
        return pred
