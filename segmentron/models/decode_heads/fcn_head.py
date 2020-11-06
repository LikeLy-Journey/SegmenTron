import torch
import torch.nn as nn
from segmentron.core.cnn import ConvModule, DepthwiseSeparableConvModule

from ..builder import HEADS
from .decode_head import BaseDecodeHead


@HEADS.register_module()
class FCNHead(BaseDecodeHead):
    """Fully Convolution Networks for Semantic Segmentation.

    This head is implemented of `FCNNet <https://arxiv.org/abs/1411.4038>`_.

    Args:
        num_convs (int): Number of convs in the head. Default: 2.
        kernel_size (int): The kernel size for convs in the head. Default: 3.
        concat_input (bool): Whether concat the input and output of convs
            before classification layer.
    """

    def __init__(self,
                 num_convs=2,
                 kernel_size=3,
                 sep_conv=False,
                 concat_input=True,
                 **kwargs):
        # assert num_convs > 0
        self.num_convs = num_convs
        self.sep_conv = sep_conv
        self.concat_input = concat_input
        self.kernel_size = kernel_size
        super(FCNHead, self).__init__(**kwargs)
        if num_convs > 0:
            convs = []
            if sep_conv:
                convs.append(
                    DepthwiseSeparableConvModule(
                        self.in_channels,
                        self.channels,
                        kernel_size=kernel_size,
                        padding=kernel_size // 2,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg))
                for i in range(num_convs - 1):
                    convs.append(
                        DepthwiseSeparableConvModule(
                            self.channels,
                            self.channels,
                            kernel_size=kernel_size,
                            padding=kernel_size // 2,
                            conv_cfg=self.conv_cfg,
                            norm_cfg=self.norm_cfg,
                            act_cfg=self.act_cfg))
            else:
                convs.append(
                    ConvModule(
                        self.in_channels,
                        self.channels,
                        kernel_size=kernel_size,
                        padding=kernel_size // 2,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg))
                for i in range(num_convs - 1):
                    convs.append(
                        ConvModule(
                            self.channels,
                            self.channels,
                            kernel_size=kernel_size,
                            padding=kernel_size // 2,
                            conv_cfg=self.conv_cfg,
                            norm_cfg=self.norm_cfg,
                            act_cfg=self.act_cfg))
            self.convs = nn.Sequential(*convs)
        if self.concat_input:
            self.conv_cat = ConvModule(
                self.in_channels + self.channels,
                self.channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg)

    def forward(self, inputs):
        """Forward function."""
        x = self._transform_inputs(inputs)

        if self.num_convs > 0:
            output = self.convs(x)
        else:
            output = x

        if self.concat_input:
            output = self.conv_cat(torch.cat([x, output], dim=1))
        output = self.cls_seg(output)
        return output
