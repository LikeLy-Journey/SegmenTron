from __future__ import division

import torch.nn.functional as F

from .segbase import SegBaseModel
from ..modules import _FCNHead
from ..config import cfg

__all__ = ['FCN', 'get_fcn']


class FCN(SegBaseModel):
    def __init__(self, nclass):
        super(FCN, self).__init__(nclass)
        self.head = _FCNHead(2048, nclass)
        if self.aux:
            self.auxlayer = _FCNHead(1024, nclass)

        self.__setattr__('decoder', ['head', 'auxlayer'] if self.aux else ['head'])

    def forward(self, x):
        size = x.size()[2:]
        _, _, c3, c4 = self.base_forward(x)

        outputs = []
        x = self.head(c4)
        x = F.interpolate(x, size, mode='bilinear', align_corners=True)
        outputs.append(x)
        if self.aux:
            auxout = self.auxlayer(c3)
            auxout = F.interpolate(auxout, size, mode='bilinear', align_corners=True)
            outputs.append(auxout)
        return tuple(outputs)


def get_fcn():
    from ..data.dataloader import datasets
    model = FCN(datasets[cfg.DATASET.NAME].NUM_CLASS)
    return model


