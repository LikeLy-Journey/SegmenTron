import torch
import torch.nn as nn
from segmentron.core.cnn import ConvModule
from segmentron.ops import resize

from ..builder import build_enhance_module


class Brick(nn.Module):

    def __init__(self,
                 in_index,
                 lateral_index=None,
                 enhance_cfg=None,
                 lateral_cfg=None,
                 upsample_cfg=None,
                 align_corners=False,
                 fusion_method='concat'):
        super(Brick, self).__init__()
        if enhance_cfg is not None:
            self.enhance_module = build_enhance_module(enhance_cfg)

        if lateral_cfg is not None:
            self.lateral_module = ConvModule(**lateral_cfg)

        self.in_index = in_index
        self.lateral_index = lateral_index
        self.fusion_method = fusion_method
        self.upsample_cfg = upsample_cfg
        self.align_corners = align_corners

    def forward(self, inputs):
        x = inputs[self.in_index]

        if self.lateral_index is not None:
            lateral_feature = inputs[self.lateral_index]

        if hasattr(self, 'enhance_module'):
            x = self.enhance_module(x)

        if hasattr(self, 'lateral_module'):
            lateral_feature = self.lateral_module(lateral_feature)

        if self.fusion_method == 'add':
            out = x + lateral_feature
        elif self.fusion_method == 'concat':
            x = resize(
                x,
                size=lateral_feature.size()[2:],
                mode='bilinear',
                align_corners=self.align_corners)
            out = torch.cat([x, lateral_feature], dim=1)
        elif self.fusion_method == 'identity':
            out = x

        return out
