"""UPerNet"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .segbase import SegBaseModel
from ..modules import _FCNHead, PyramidPooling
from .model_zoo import MODEL_REGISTRY
from ..config import cfg

__all__ = ['UPerNet']


@MODEL_REGISTRY.register()
class UPerNet(SegBaseModel):
    r"""Unified Perceptual Parsing for Scene Understanding
    Reference:
        T. Xiao, Y. Liu, B. Zhou, Y. Jiang, and J. Sun,
        “Unified Perceptual Parsing for Scene Understanding,”
        in Computer Vision – ECCV 2018,Lecture Notes in Computer Science,
        2018, pp. 432–448. doi: 10.1007/978-3-030-01228-1_26.
    """

    def __init__(self):
        super(UPerNet, self).__init__()
        self.in_channels = cfg.MODEL.UPERNET.IN_CHANNELS
        self.channels = cfg.MODEL.UPERNET.CHANNELS
        self.pool_scales = cfg.MODEL.UPERNET.POOL_SCALES
        self.head = _UPerNetHead(self.nclass, self.in_channels, self.channels, self.pool_scales)
        if self.aux:
            self.auxlayer = _FCNHead(self.in_channels[2], self.nclass, self.channels)

        self.__setattr__('decoder', ['head', 'auxlayer'] if self.aux else ['head'])

    def forward(self, x):
        size = x.size()[2:]
        c1, c2, c3, c4 = self.encoder(x)
        outputs = []
        x = self.head((c1, c2, c3, c4))
        x = F.interpolate(x, size, mode='bilinear', align_corners=True)
        outputs.append(x)

        if self.aux:
            auxout = self.auxlayer(c3)
            auxout = F.interpolate(auxout, size, mode='bilinear', align_corners=True)
            outputs.append(auxout)
        return {"inference_results": x, "loss_results": tuple(outputs)}


class _UPerNetHead(nn.Module):
    def __init__(self,
                 nclass,
                 in_channels=(96, 192, 384, 768),
                 channels=512,
                 pool_scales=(1, 2, 3, 6),
                 norm_layer=nn.BatchNorm2d,
                 norm_kwargs=None,
                 **kwargs):
        super(_UPerNetHead, self).__init__()
        self.in_channels = in_channels
        self.channels = channels
        # PSP Module
        self.psp = PyramidPooling(self.in_channels[-1], self.channels, sizes=pool_scales, norm_layer=norm_layer)
        # PSP Residual Connection
        self.bottleneck = nn.Sequential(
            nn.Conv2d(self.in_channels[-1] + len(pool_scales) * channels, channels, 3, padding=1, bias=False),
            norm_layer(512, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(inplace=True),
        )
        # FPN Module
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        for in_channels in self.in_channels[:-1]:  # skip the top layer
            l_conv = nn.Sequential(
                nn.Conv2d(in_channels, channels, 1, bias=False),
                norm_layer(channels, **({} if norm_kwargs is None else norm_kwargs)),
                nn.ReLU(inplace=False),
            )
            fpn_conv = nn.Sequential(
                nn.Conv2d(channels, channels, 3, padding=1, bias=False),
                norm_layer(channels, **({} if norm_kwargs is None else norm_kwargs)),
                nn.ReLU(inplace=False),
            )
            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)
        # FPN Residual Connection
        self.bottleneck_fpn = nn.Sequential(
            nn.Conv2d(channels * len(self.in_channels), channels, 3, padding=1, bias=False),
            norm_layer(channels, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(inplace=True),
        )
        # Class Segmentor
        self.block = nn.Sequential(
            nn.Dropout2d(0.1),
            nn.Conv2d(channels, nclass, 1)
        )

    def forward(self, inputs):
        # PSP forward
        psp_out = self.psp(inputs[-1])
        psp_out = self.bottleneck(psp_out)
        # build laterals
        laterals = [lateral_conv(inputs[i]) for i, lateral_conv in enumerate(self.lateral_convs)]
        laterals.append(psp_out)
        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[2:]
            laterals[i - 1] = laterals[i - 1] + F.interpolate(laterals[i], size=prev_shape, mode='bilinear',
                                                              align_corners=False)
        # build outputs
        fpn_out = [fpn_conv(laterals[i]) for i, fpn_conv in enumerate(self.fpn_convs)]
        fpn_out.append(psp_out)
        for i in range(used_backbone_levels - 1, 0, -1):
            fpn_out[i] = F.interpolate(fpn_out[i], size=fpn_out[0].shape[2:], mode='bilinear', align_corners=False)
        fpn_out = torch.cat(fpn_out, dim=1)
        fpn_out = self.bottleneck_fpn(fpn_out)
        # class segmentor
        out = self.block(fpn_out)
        return out
