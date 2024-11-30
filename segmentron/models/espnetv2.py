"ESPNetv2: A Light-weight, Power Efficient, and General Purpose for Semantic Segmentation"
import torch
import torch.nn as nn
import torch.nn.functional as F

from .segbase import SegBaseModel
from .model_zoo import MODEL_REGISTRY
from ..modules import _ConvBNPReLU, EESP, _BNPReLU, _FCNHead
from ..config import cfg


@MODEL_REGISTRY.register()
class ESPNetV2(SegBaseModel):
    r"""ESPNetV2
    Reference:
        Sachin Mehta, et al. "ESPNetv2: A Light-weight, Power Efficient, and General Purpose Convolutional Neural Network."
        arXiv preprint arXiv:1811.11431 (2018).
    """

    def __init__(self, **kwargs):
        super(ESPNetV2, self).__init__()
        self.proj_L4_C = _ConvBNPReLU(256, 128, 1, **kwargs)
        self.pspMod = nn.Sequential(
            EESP(256, 128, stride=1, k=4, r_lim=7, **kwargs),
            _PSPModule(128, 128, **kwargs))
        self.project_l3 = nn.Sequential(
            nn.Dropout2d(0.1),
            nn.Conv2d(128, self.nclass, 1, bias=False))
        self.act_l3 = _BNPReLU(self.nclass, **kwargs)
        self.project_l2 = _ConvBNPReLU(64 + self.nclass, self.nclass, 1, **kwargs)
        self.project_l1 = nn.Sequential(
            nn.Dropout2d(0.1),
            nn.Conv2d(32 + self.nclass, self.nclass, 1, bias=False))

        self.__setattr__('exclusive', ['proj_L4_C', 'pspMod', 'project_l3', 'act_l3', 'project_l2', 'project_l1'])

    def forward(self, x):
        size = x.size()[2:]
        out_l1, out_l2, out_l3, out_l4 = self.encoder(x, seg=True)
        out_l4_proj = self.proj_L4_C(out_l4)
        up_l4_to_l3 = F.interpolate(out_l4_proj, scale_factor=2, mode='bilinear', align_corners=True)
        merged_l3_upl4 = self.pspMod(torch.cat([out_l3, up_l4_to_l3], 1))
        proj_merge_l3_bef_act = self.project_l3(merged_l3_upl4)
        proj_merge_l3 = self.act_l3(proj_merge_l3_bef_act)
        out_up_l3 = F.interpolate(proj_merge_l3, scale_factor=2, mode='bilinear', align_corners=True)
        merge_l2 = self.project_l2(torch.cat([out_l2, out_up_l3], 1))
        out_up_l2 = F.interpolate(merge_l2, scale_factor=2, mode='bilinear', align_corners=True)
        merge_l1 = self.project_l1(torch.cat([out_l1, out_up_l2], 1))

        outputs = list()
        merge1_l1 = F.interpolate(merge_l1, scale_factor=2, mode='bilinear', align_corners=True)
        outputs.append(merge1_l1)
        if self.aux:
            # different from paper
            auxout = F.interpolate(proj_merge_l3_bef_act, size, mode='bilinear', align_corners=True)
            outputs.append(auxout)

        return {"inference_results": x, "loss_results": tuple(outputs)}


# different from PSPNet
class _PSPModule(nn.Module):
    def __init__(self, in_channels, out_channels=1024, sizes=(1, 2, 4, 8), **kwargs):
        super(_PSPModule, self).__init__()
        self.stages = nn.ModuleList(
            [nn.Conv2d(in_channels, in_channels, 3, 1, 1, groups=in_channels, bias=False) for _ in sizes])
        self.project = _ConvBNPReLU(in_channels * (len(sizes) + 1), out_channels, 1, 1, **kwargs)

    def forward(self, x):
        size = x.size()[2:]
        feats = [x]
        for stage in self.stages:
            x = F.avg_pool2d(x, kernel_size=3, stride=2, padding=1)
            upsampled = F.interpolate(stage(x), size, mode='bilinear', align_corners=True)
            feats.append(upsampled)
        return self.project(torch.cat(feats, dim=1))
