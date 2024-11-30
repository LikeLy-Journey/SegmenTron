import torch
import torch.nn as nn
import torch.nn.functional as F

from .segbase import SegBaseModel
from .model_zoo import MODEL_REGISTRY
from ..modules import _FCNHead

__all__ = ['RefineNet']


@MODEL_REGISTRY.register()
class RefineNet(SegBaseModel):

    def __init__(self):
        super(RefineNet, self).__init__()
        self.head = _RefineHead(self.nclass, norm_layer=self.norm_layer)
        if self.aux:
            self.auxlayer = _FCNHead(728, self.nclass)
        self.__setattr__('decoder', ['head', 'auxlayer'] if self.aux else ['head'])

    def forward(self, x):
        size = x.size()[2:]
        c1, c2, c3, c4 = self.encoder(x)

        outputs = list()
        x = self.head(c1, c2, c3, c4)
        x = F.interpolate(x, size, mode='bilinear', align_corners=True)

        outputs.append(x)
        if self.aux:
            auxout = self.auxlayer(c3)
            auxout = F.interpolate(auxout, size, mode='bilinear', align_corners=True)
            outputs.append(auxout)
        return {"inference_results": x, "loss_results": tuple(outputs)}


class _RefineHead(nn.Module):
    def __init__(self, nclass, norm_layer=nn.BatchNorm2d):
        super(_RefineHead, self).__init__()
        self.do = nn.Dropout(p=0.5)
        self.relu = nn.ReLU(inplace=True)
        self.p_ims1d2_outl1_dimred = nn.Conv2d(2048, 512, 1, bias=False)
        self.mflow_conv_g1_pool = self._make_crp(512, 512, 4)
        self.mflow_conv_g1_b3_joint_varout_dimred = nn.Conv2d(512, 256, 1, bias=False)
        self.p_ims1d2_outl2_dimred = nn.Conv2d(1024, 256, 1, bias=False)
        self.adapt_stage2_b2_joint_varout_dimred = nn.Conv2d(256, 256, 1, bias=False)
        self.mflow_conv_g2_pool = self._make_crp(256, 256, 4)
        self.mflow_conv_g2_b3_joint_varout_dimred = nn.Conv2d(256, 256, 1, bias=False)

        self.p_ims1d2_outl3_dimred = nn.Conv2d(512, 256, 1, bias=False)
        self.adapt_stage3_b2_joint_varout_dimred = nn.Conv2d(256, 256, 1, bias=False)
        self.mflow_conv_g3_pool = self._make_crp(256, 256, 4)
        self.mflow_conv_g3_b3_joint_varout_dimred = nn.Conv2d(256, 256, 1, bias=False)

        self.p_ims1d2_outl4_dimred = nn.Conv2d(256, 256, 1, bias=False)
        self.adapt_stage4_b2_joint_varout_dimred = nn.Conv2d(256, 256, 1, bias=False)
        self.mflow_conv_g4_pool = self._make_crp(256, 256, 4)

        self.clf_conv = nn.Conv2d(256, nclass, kernel_size=3, stride=1,
                                  padding=1, bias=True)

    def _make_crp(self, in_planes, out_planes, stages):
        layers = [CRPBlock(in_planes, out_planes, stages)]
        return nn.Sequential(*layers)

    def forward(self, l1, l2, l3, l4):
        l4 = self.do(l4)
        l3 = self.do(l3)

        x4 = self.p_ims1d2_outl1_dimred(l4)
        x4 = self.relu(x4)
        x4 = self.mflow_conv_g1_pool(x4)
        x4 = self.mflow_conv_g1_b3_joint_varout_dimred(x4)
        x4 = F.interpolate(x4, size=l3.size()[2:], mode='bilinear', align_corners=True)

        x3 = self.p_ims1d2_outl2_dimred(l3)
        x3 = self.adapt_stage2_b2_joint_varout_dimred(x3)
        x3 = x3 + x4
        x3 = F.relu(x3)
        x3 = self.mflow_conv_g2_pool(x3)
        x3 = self.mflow_conv_g2_b3_joint_varout_dimred(x3)
        x3 = F.interpolate(x3, size=l2.size()[2:], mode='bilinear', align_corners=True)

        x2 = self.p_ims1d2_outl3_dimred(l2)
        x2 = self.adapt_stage3_b2_joint_varout_dimred(x2)
        x2 = x2 + x3
        x2 = F.relu(x2)
        x2 = self.mflow_conv_g3_pool(x2)
        x2 = self.mflow_conv_g3_b3_joint_varout_dimred(x2)
        x2 = F.interpolate(x2, size=l1.size()[2:], mode='bilinear', align_corners=True)

        x1 = self.p_ims1d2_outl4_dimred(l1)
        x1 = self.adapt_stage4_b2_joint_varout_dimred(x1)
        x1 = x1 + x2
        x1 = F.relu(x1)
        x1 = self.mflow_conv_g4_pool(x1)

        out = self.clf_conv(x1)
        return out


class CRPBlock(nn.Module):

    def __init__(self, in_planes, out_planes, n_stages):
        super(CRPBlock, self).__init__()
        for i in range(n_stages):
            setattr(self, '{}_{}'.format(i + 1, 'outvar_dimred'),
                    nn.Conv2d(in_planes if (i == 0) else out_planes,
                            out_planes, 1, stride=1,
                            bias=False))
        self.stride = 1
        self.n_stages = n_stages
        self.maxpool = nn.MaxPool2d(kernel_size=5, stride=1, padding=2)

    def forward(self, x):
        top = x
        for i in range(self.n_stages):
            top = self.maxpool(top)
            top = getattr(self, '{}_{}'.format(i + 1, 'outvar_dimred'))(top)
            x = top + x
        return x