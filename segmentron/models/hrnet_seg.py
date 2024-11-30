# this code is heavily based on https://github.com/HRNet

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch._utils
import torch.nn.functional as F

from .segbase import SegBaseModel
from .model_zoo import MODEL_REGISTRY
from ..config import cfg


@MODEL_REGISTRY.register(name='HRNet')
class HighResolutionNet(SegBaseModel):
    def __init__(self):
        super(HighResolutionNet, self).__init__()
        self.hrnet_head = _HRNetHead(self.nclass, self.encoder.last_inp_channels)
        self.__setattr__('decoder', ['hrnet_head'])

    def forward(self, x):
        shape = x.shape[2:]
        x = self.encoder(x)
        x = self.hrnet_head(x)
        x = F.interpolate(x, size=shape, mode='bilinear', align_corners=False)
        return {"inference_results": x, "loss_results": tuple([x])}


class _HRNetHead(nn.Module):
    def __init__(self, nclass, last_inp_channels, norm_layer=nn.BatchNorm2d):
        super(_HRNetHead, self).__init__()

        self.last_layer = nn.Sequential(
            nn.Conv2d(
                in_channels=last_inp_channels,
                out_channels=last_inp_channels,
                kernel_size=1,
                stride=1,
                padding=0),

            norm_layer(last_inp_channels),
            nn.ReLU(inplace=False),
            nn.Conv2d(
                in_channels=last_inp_channels,
                out_channels=nclass,
                kernel_size=cfg.MODEL.HRNET.FINAL_CONV_KERNEL,
                stride=1,
                padding=1 if cfg.MODEL.HRNET.FINAL_CONV_KERNEL == 3 else 0)
        )

    def forward(self, x):
        # Upsampling
        x0_h, x0_w = x[0].size(2), x[0].size(3)
        x1 = F.interpolate(x[1], size=(x0_h, x0_w), mode='bilinear', align_corners=False)
        x2 = F.interpolate(x[2], size=(x0_h, x0_w), mode='bilinear', align_corners=False)
        x3 = F.interpolate(x[3], size=(x0_h, x0_w), mode='bilinear', align_corners=False)

        x = torch.cat([x[0], x1, x2, x3], 1)
        x = self.last_layer(x)
        return x
