import torch
import torch.nn as nn
import torch.nn.functional as F

from .segbase import SegBaseModel
from .model_zoo import MODEL_REGISTRY
from ..modules import _ConvBNReLU, SeparableConv2d, _ASPP, _FCNHead
from ..config import cfg

__all__ = ['DeepLabV3Plus']


@MODEL_REGISTRY.register(name='DeepLabV3_Plus')
class DeepLabV3Plus(SegBaseModel):
    r"""DeepLabV3Plus
    Reference:
        Chen, Liang-Chieh, et al. "Encoder-Decoder with Atrous Separable Convolution for Semantic
        Image Segmentation."
    """
    def __init__(self):
        super(DeepLabV3Plus, self).__init__()
        if self.backbone.startswith('mobilenet'):
            c1_channels = 24
            c4_channels = 320
        else:
            c1_channels = 256
            c4_channels = 2048
        self.head = _DeepLabHead(self.nclass, c1_channels=c1_channels, c4_channels=c4_channels)
        if self.aux:
            self.auxlayer = _FCNHead(728, self.nclass)
        self.__setattr__('decoder', ['head', 'auxlayer'] if self.aux else ['head'])

    def forward(self, x):
        size = x.size()[2:]
        c1, _, c3, c4 = self.encoder(x)

        outputs = list()
        x = self.head(c4, c1)
        x = F.interpolate(x, size, mode='bilinear', align_corners=True)

        outputs.append(x)
        if self.aux:
            auxout = self.auxlayer(c3)
            auxout = F.interpolate(auxout, size, mode='bilinear', align_corners=True)
            outputs.append(auxout)
        return {"inference_results": x, "loss_results": tuple(outputs)}


class _DeepLabHead(nn.Module):
    def __init__(self, nclass, c1_channels=256, c4_channels=2048, norm_layer=nn.BatchNorm2d):
        super(_DeepLabHead, self).__init__()
        self.use_aspp = cfg.MODEL.DEEPLABV3_PLUS.USE_ASPP
        self.use_decoder = cfg.MODEL.DEEPLABV3_PLUS.ENABLE_DECODER
        last_channels = c4_channels
        if self.use_aspp:
            self.aspp = _ASPP(c4_channels, 256)
            last_channels = 256
        if self.use_decoder:
            self.c1_block = _ConvBNReLU(c1_channels, 48, 1, norm_layer=norm_layer)
            last_channels += 48
        self.block = nn.Sequential(
            SeparableConv2d(last_channels, 256, 3, norm_layer=norm_layer, relu_first=False),
            SeparableConv2d(256, 256, 3, norm_layer=norm_layer, relu_first=False),
            nn.Conv2d(256, nclass, 1))

    def forward(self, x, c1):
        size = c1.size()[2:]
        if self.use_aspp:
            x = self.aspp(x)
        if self.use_decoder:
            x = F.interpolate(x, size, mode='bilinear', align_corners=True)
            c1 = self.c1_block(c1)
            return self.block(torch.cat([x, c1], dim=1))

        return self.block(x)
