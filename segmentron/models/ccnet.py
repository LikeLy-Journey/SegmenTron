import torch
import torch.nn as nn
import torch.nn.functional as F

from .segbase import SegBaseModel
from .model_zoo import MODEL_REGISTRY
from ..modules import _FCNHead
from ..modules.cc_attention import CrissCrossAttention
from ..config import cfg


@MODEL_REGISTRY.register()
class CCNet(SegBaseModel):
    r"""CCNet
    Reference:
        Zilong Huang, et al. "CCNet: Criss-Cross Attention for Semantic Segmentation."
        arXiv preprint arXiv:1811.11721 (2018).
    """

    def __init__(self):
        super(CCNet, self).__init__()
        self.head = _CCHead(self.nclass, norm_layer=self.norm_layer)
        if self.aux:
            self.auxlayer = _FCNHead(1024, self.nclass, norm_layer=self.norm_layer)

        self.__setattr__('decoder', ['head', 'auxlayer'] if self.aux else ['head'])

    def forward(self, x):
        size = x.size()[2:]
        _, _, c3, c4 = self.base_forward(x)
        outputs = list()
        x = self.head(c4)
        x = F.interpolate(x, size, mode='bilinear', align_corners=True)
        outputs.append(x)

        if self.aux:
            auxout = self.auxlayer(c3)
            auxout = F.interpolate(auxout, size, mode='bilinear', align_corners=True)
            outputs.append(auxout)
        return {"inference_results": x, "loss_results": tuple(outputs)}


class _CCHead(nn.Module):
    def __init__(self, nclass, norm_layer=nn.BatchNorm2d):
        super(_CCHead, self).__init__()
        self.rcca = _RCCAModule(2048, 512, norm_layer)
        self.out = nn.Conv2d(512, nclass, 1)

    def forward(self, x):
        x = self.rcca(x)
        x = self.out(x)
        return x


class _RCCAModule(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer):
        super(_RCCAModule, self).__init__()
        self.recurrence = cfg.MODEL.CCNET.RECURRENCE
        inter_channels = in_channels // 4
        self.conva = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
            norm_layer(inter_channels),
            nn.ReLU(True))
        self.cca = CrissCrossAttention(inter_channels)
        self.convb = nn.Sequential(
            nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
            norm_layer(inter_channels),
            nn.ReLU(True))

        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels + inter_channels, out_channels, 3, padding=1, bias=False),
            norm_layer(out_channels),
            nn.Dropout2d(0.1))

    def forward(self, x):
        out = self.conva(x)
        for i in range(self.recurrence):
            out = self.cca(out)
        out = self.convb(out)
        out = torch.cat([x, out], dim=1)
        out = self.bottleneck(out)

        return out
