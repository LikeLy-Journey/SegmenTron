""" Deep Feature Aggregation"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .segbase import SegBaseModel
from .backbones.xception import Enc, FCAttention
from .model_zoo import MODEL_REGISTRY
from ..modules import _ConvBNReLU

__all__ = ['DFANet']


@MODEL_REGISTRY.register()
class DFANet(SegBaseModel):
    def __init__(self, **kwargs):
        super(DFANet, self).__init__()

        self.enc2_2 = Enc(240, 48, 4, **kwargs)
        self.enc3_2 = Enc(144, 96, 6, **kwargs)
        self.enc4_2 = Enc(288, 192, 4, **kwargs)
        self.fca_2 = FCAttention(192, **kwargs)

        self.enc2_3 = Enc(240, 48, 4, **kwargs)
        self.enc3_3 = Enc(144, 96, 6, **kwargs)
        self.enc3_4 = Enc(288, 192, 4, **kwargs)
        self.fca_3 = FCAttention(192, **kwargs)

        self.enc2_1_reduce = _ConvBNReLU(48, 32, 1, **kwargs)
        self.enc2_2_reduce = _ConvBNReLU(48, 32, 1, **kwargs)
        self.enc2_3_reduce = _ConvBNReLU(48, 32, 1, **kwargs)
        self.conv_fusion = _ConvBNReLU(32, 32, 1, **kwargs)

        self.fca_1_reduce = _ConvBNReLU(192, 32, 1, **kwargs)
        self.fca_2_reduce = _ConvBNReLU(192, 32, 1, **kwargs)
        self.fca_3_reduce = _ConvBNReLU(192, 32, 1, **kwargs)
        self.conv_out = nn.Conv2d(32, self.nclass, 1)

        self.__setattr__('decoder', ['enc2_2', 'enc3_2', 'enc4_2', 'fca_2', 'enc2_3', 'enc3_3', 'enc3_4',
                                     'fca_3', 'enc2_1_reduce', 'enc2_2_reduce', 'enc2_3_reduce', 'conv_fusion',
                                     'fca_1_reduce', 'fca_2_reduce', 'fca_3_reduce', 'conv_out'])

    def forward(self, x):
        # backbone
        stage1_conv1 = self.encoder.conv1(x)
        stage1_enc2 = self.encoder.enc2(stage1_conv1)
        stage1_enc3 = self.encoder.enc3(stage1_enc2)
        stage1_enc4 = self.encoder.enc4(stage1_enc3)
        stage1_fca = self.encoder.fca(stage1_enc4)
        stage1_out = F.interpolate(stage1_fca, scale_factor=4, mode='bilinear', align_corners=True)

        # stage2
        stage2_enc2 = self.enc2_2(torch.cat([stage1_enc2, stage1_out], dim=1))
        stage2_enc3 = self.enc3_2(torch.cat([stage1_enc3, stage2_enc2], dim=1))
        stage2_enc4 = self.enc4_2(torch.cat([stage1_enc4, stage2_enc3], dim=1))
        stage2_fca = self.fca_2(stage2_enc4)
        stage2_out = F.interpolate(stage2_fca, scale_factor=4, mode='bilinear', align_corners=True)

        # stage3
        stage3_enc2 = self.enc2_3(torch.cat([stage2_enc2, stage2_out], dim=1))
        stage3_enc3 = self.enc3_3(torch.cat([stage2_enc3, stage3_enc2], dim=1))
        stage3_enc4 = self.enc3_4(torch.cat([stage2_enc4, stage3_enc3], dim=1))
        stage3_fca = self.fca_3(stage3_enc4)

        stage1_enc2_decoder = self.enc2_1_reduce(stage1_enc2)
        stage2_enc2_docoder = F.interpolate(self.enc2_2_reduce(stage2_enc2), scale_factor=2,
                                            mode='bilinear', align_corners=True)
        stage3_enc2_decoder = F.interpolate(self.enc2_3_reduce(stage3_enc2), scale_factor=4,
                                            mode='bilinear', align_corners=True)
        fusion = stage1_enc2_decoder + stage2_enc2_docoder + stage3_enc2_decoder
        fusion = self.conv_fusion(fusion)

        stage1_fca_decoder = F.interpolate(self.fca_1_reduce(stage1_fca), scale_factor=4,
                                           mode='bilinear', align_corners=True)
        stage2_fca_decoder = F.interpolate(self.fca_2_reduce(stage2_fca), scale_factor=8,
                                           mode='bilinear', align_corners=True)
        stage3_fca_decoder = F.interpolate(self.fca_3_reduce(stage3_fca), scale_factor=16,
                                           mode='bilinear', align_corners=True)
        fusion = fusion + stage1_fca_decoder + stage2_fca_decoder + stage3_fca_decoder

        outputs = list()
        out = self.conv_out(fusion)
        out = F.interpolate(out, scale_factor=4, mode='bilinear', align_corners=True)
        outputs.append(out)

        return {"inference_results": x, "loss_results": tuple(outputs)}
