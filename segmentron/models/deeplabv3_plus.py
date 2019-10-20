import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbones import get_xception, get_resnet
from ..modules import _ConvBNReLU, SeparableConv2d, _ASPP, _FCNHead
from ..utils.config import cfg
__all__ = ['DeepLabV3Plus', 'get_deeplabv3_plus', 'get_deeplabv3_plus_xception_voc']


class DeepLabV3Plus(nn.Module):
    r"""DeepLabV3Plus
    Parameters
    ----------
    nclass : int
        Number of categories for the training dataset.

    """

    def __init__(self, nclass):
        super(DeepLabV3Plus, self).__init__()
        self.aux = cfg.SOLVER.AUX
        self.nclass = nclass

        backbone = cfg.MODEL.BACKBONE.lower()
        if backbone.startswith('xception'):
            self.encoder = get_xception(backbone)
        elif backbone.startswith('resnet'):
            self.encoder = get_resnet(backbone)
        else:
            raise RuntimeError('unknown backbone: {}'.format(backbone))

        self.head = _DeepLabHead(nclass)
        if self.aux:
            self.auxlayer = _FCNHead(728, nclass)
        self.__setattr__('decoder', ['head', 'auxlayer'] if self.aux else ['head'])

    def forward(self, x):
        size = x.size()[2:]
        c1, c3, c4 = self.encoder(x)

        outputs = list()
        x = self.head(c4, c1)
        x = F.interpolate(x, size, mode='bilinear', align_corners=True)

        outputs.append(x)
        if self.aux:
            auxout = self.auxlayer(c3)
            auxout = F.interpolate(auxout, size, mode='bilinear', align_corners=True)
            outputs.append(auxout)
        return tuple(outputs)


class _DeepLabHead(nn.Module):
    def __init__(self, nclass, c1_channels=256, norm_layer=nn.BatchNorm2d):
        super(_DeepLabHead, self).__init__()
        self.aspp = _ASPP(2048, c1_channels)
        self.c1_block = _ConvBNReLU(c1_channels, 48, 1, norm_layer=norm_layer)
        self.block = nn.Sequential(
            SeparableConv2d(304, 256, 3, norm_layer=norm_layer, relu_first=False),
            SeparableConv2d(256, 256, 3, norm_layer=norm_layer, relu_first=False),
            nn.Conv2d(256, nclass, 1))

    def forward(self, x, c1):
        size = c1.size()[2:]
        c1 = self.c1_block(c1)
        x = self.aspp(x)
        x = F.interpolate(x, size, mode='bilinear', align_corners=True)
        return self.block(torch.cat([x, c1], dim=1))


def get_deeplabv3_plus():
    from ..data.dataloader import datasets
    model = DeepLabV3Plus(datasets[cfg.DATASET].NUM_CLASS)
    return model


def get_deeplabv3_plus_xception_voc(**kwargs):
    return get_deeplabv3_plus('pascal_voc', 'xception', **kwargs)


if __name__ == '__main__':
    model = get_deeplabv3_plus_xception_voc()
