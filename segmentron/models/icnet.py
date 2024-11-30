"""Image Cascade Network"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .segbase import SegBaseModel
from .model_zoo import MODEL_REGISTRY
from ..modules.basic import _ConvBNReLU
from ..config import cfg

__all__ = ['ICNet']


@MODEL_REGISTRY.register()
class ICNet(SegBaseModel):
    """Image Cascade Network"""

    def __init__(self):
        super(ICNet, self).__init__()
        self.conv_sub1 = nn.Sequential(
            _ConvBNReLU(3, 32, 3, 2),
            _ConvBNReLU(32, 32, 3, 2),
            _ConvBNReLU(32, 64, 3, 2)
        )

        self.head = _ICHead(self.nclass)
        self.__setattr__('decoder', ['conv_sub1', 'head'])

    def forward(self, x):
        size = x.size()[2:]
        # sub 1
        x_sub1 = self.conv_sub1(x)

        # sub 2
        x_sub2 = F.interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=True)
        _, x_sub2, _, _ = self.encoder(x_sub2)

        # sub 4
        x_sub4 = F.interpolate(x, scale_factor=0.25, mode='bilinear', align_corners=True)
        _, _, _, x_sub4 = self.encoder(x_sub4)

        outputs = self.head(x_sub1, x_sub2, x_sub4, size)

        return {"inference_results": x, "loss_results": tuple(outputs)}


class _ICHead(nn.Module):
    def __init__(self, nclass, norm_layer=nn.BatchNorm2d):
        super(_ICHead, self).__init__()
        scale = cfg.MODEL.BACKBONE_SCALE
        self.cff_12 = CascadeFeatureFusion(int(512 * scale), 64, 128, nclass, norm_layer)
        self.cff_24 = CascadeFeatureFusion(int(2048 * scale), int(512 * scale), 128, nclass, norm_layer)
        self.conv_cls = nn.Conv2d(128, nclass, 1, bias=False)

    def forward(self, x_sub1, x_sub2, x_sub4, size):
        outputs = list()
        x_cff_24, x_24_cls = self.cff_24(x_sub4, x_sub2)
        outputs.append(x_24_cls)
        x_cff_12, x_12_cls = self.cff_12(x_sub2, x_sub1)
        outputs.append(x_12_cls)

        up_x2 = F.interpolate(x_cff_12, scale_factor=2, mode='bilinear', align_corners=True)
        up_x2 = self.conv_cls(up_x2)
        outputs.append(up_x2)

        up_x8 = F.interpolate(up_x2, size, mode='bilinear', align_corners=True)
        outputs.append(up_x8)
        # 1 -> 1/4 -> 1/8 -> 1/16
        outputs.reverse()

        return outputs


class CascadeFeatureFusion(nn.Module):
    """CFF Unit"""

    def __init__(self, low_channels, high_channels, out_channels, nclass, norm_layer=nn.BatchNorm2d):
        super(CascadeFeatureFusion, self).__init__()
        self.conv_low = nn.Sequential(
            nn.Conv2d(low_channels, out_channels, 3, padding=2, dilation=2, bias=False),
            norm_layer(out_channels)
        )
        self.conv_high = nn.Sequential(
            nn.Conv2d(high_channels, out_channels, 1, bias=False),
            norm_layer(out_channels)
        )
        self.conv_low_cls = nn.Conv2d(out_channels, nclass, 1, bias=False)

    def forward(self, x_low, x_high):
        x_low = F.interpolate(x_low, size=x_high.size()[2:], mode='bilinear', align_corners=True)
        x_low = self.conv_low(x_low)
        x_high = self.conv_high(x_high)
        x = x_low + x_high
        x = F.relu(x, inplace=True)
        x_low_cls = self.conv_low_cls(x_low)

        return x, x_low_cls
