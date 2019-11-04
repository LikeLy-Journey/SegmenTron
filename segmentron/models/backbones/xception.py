import os
import logging
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo

from collections import OrderedDict
from ...modules import SeparableConv2d
from ...utils.config import cfg

__all__ = ['Xception65', 'get_xception']

model_urls = {
    'xception65': 'https://github.com/LikeLy-Journey/SegmenTron/releases/download/v0.1.0/tf-xception65-270e81cf.pth',
}

class XceptionBlock(nn.Module):
    def __init__(self, channel_list, stride=1, dilation=1, skip_connection_type='conv', relu_first=True, low_feat=False):
        super().__init__()

        assert len(channel_list) == 4
        self.skip_connection_type = skip_connection_type
        self.relu_first = relu_first
        self.low_feat = low_feat

        if self.skip_connection_type == 'conv':
            self.conv = nn.Conv2d(channel_list[0], channel_list[-1], 1, stride=stride, bias=False)
            self.bn = nn.BatchNorm2d(channel_list[-1])

        self.sep_conv1 = SeparableConv2d(channel_list[0], channel_list[1],
                                         dilation=dilation, relu_first=relu_first)
        self.sep_conv2 = SeparableConv2d(channel_list[1], channel_list[2],
                                         dilation=dilation, relu_first=relu_first)
        self.sep_conv3 = SeparableConv2d(channel_list[2], channel_list[3],
                                         dilation=dilation, relu_first=relu_first, stride=stride)

    def forward(self, inputs):
        sc1 = self.sep_conv1(inputs)
        sc2 = self.sep_conv2(sc1)
        residual = self.sep_conv3(sc2)

        if self.skip_connection_type == 'conv':
            shortcut = self.conv(inputs)
            shortcut = self.bn(shortcut)
            outputs = residual + shortcut
        elif self.skip_connection_type == 'sum':
            outputs = residual + inputs
        elif self.skip_connection_type == 'none':
            outputs = residual
        else:
            raise ValueError('Unsupported skip connection type.')

        if self.low_feat:
            return outputs, sc2
        else:
            return outputs

class Xception65(nn.Module):
    def __init__(self, output_stride=16):
        super().__init__()

        if output_stride == 16:
            entry_block3_stride = 2
            middle_block_dilation = 1
            exit_block_dilations = (1, 2)
        elif output_stride == 8:
            entry_block3_stride = 1
            middle_block_dilation = 2
            exit_block_dilations = (2, 4)
        else:
            raise NotImplementedError

        # Entry flow
        self.conv1 = nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU()

        self.conv2 = nn.Conv2d(32, 64, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)

        self.block1 = XceptionBlock([64, 128, 128, 128], stride=2)
        self.block2 = XceptionBlock([128, 256, 256, 256], stride=2, low_feat=True)
        self.block3 = XceptionBlock([256, 728, 728, 728], stride=entry_block3_stride)

        # Middle flow (16 units)
        self.block4 = XceptionBlock([728, 728, 728, 728], dilation=middle_block_dilation, skip_connection_type='sum')
        self.block5 = XceptionBlock([728, 728, 728, 728], dilation=middle_block_dilation, skip_connection_type='sum')
        self.block6 = XceptionBlock([728, 728, 728, 728], dilation=middle_block_dilation, skip_connection_type='sum')
        self.block7 = XceptionBlock([728, 728, 728, 728], dilation=middle_block_dilation, skip_connection_type='sum')
        self.block8 = XceptionBlock([728, 728, 728, 728], dilation=middle_block_dilation, skip_connection_type='sum')
        self.block9 = XceptionBlock([728, 728, 728, 728], dilation=middle_block_dilation, skip_connection_type='sum')
        self.block10 = XceptionBlock([728, 728, 728, 728], dilation=middle_block_dilation, skip_connection_type='sum')
        self.block11 = XceptionBlock([728, 728, 728, 728], dilation=middle_block_dilation, skip_connection_type='sum')
        self.block12 = XceptionBlock([728, 728, 728, 728], dilation=middle_block_dilation, skip_connection_type='sum')
        self.block13 = XceptionBlock([728, 728, 728, 728], dilation=middle_block_dilation, skip_connection_type='sum')
        self.block14 = XceptionBlock([728, 728, 728, 728], dilation=middle_block_dilation, skip_connection_type='sum')
        self.block15 = XceptionBlock([728, 728, 728, 728], dilation=middle_block_dilation, skip_connection_type='sum')
        self.block16 = XceptionBlock([728, 728, 728, 728], dilation=middle_block_dilation, skip_connection_type='sum')
        self.block17 = XceptionBlock([728, 728, 728, 728], dilation=middle_block_dilation, skip_connection_type='sum')
        self.block18 = XceptionBlock([728, 728, 728, 728], dilation=middle_block_dilation, skip_connection_type='sum')
        self.block19 = XceptionBlock([728, 728, 728, 728], dilation=middle_block_dilation, skip_connection_type='sum')

        # Exit flow
        self.block20 = XceptionBlock([728, 728, 1024, 1024], dilation=exit_block_dilations[0])
        self.block21 = XceptionBlock([1024, 1536, 1536, 2048], dilation=exit_block_dilations[1],
                                     skip_connection_type='none', relu_first=False)

    def forward(self, x):
        # Entry flow
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.block1(x)
        x, low_level_feat = self.block2(x)  # b, h//4, w//4, 256
        x = self.block3(x)  # b, h//8, w//8, 728

        # Middle flow
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)
        x = self.block11(x)
        x = self.block12(x)
        x = self.block13(x)
        x = self.block14(x)
        x = self.block15(x)
        x = self.block16(x)
        x = self.block17(x)
        x = self.block18(x)
        x = self.block19(x)
        mid_level_feat = x
        # Exit flow
        x = self.block20(x)
        x = self.block21(x)

        return low_level_feat, mid_level_feat, x

# Constructor
def get_xception(backbone):
    model = Xception65(cfg.MODEL.OUTPUT_STRIDE)
    if cfg.PHASE == 'train' and cfg.TRAIN.BACKBONE_PRETRAINED and (not cfg.TRAIN.PRETRAINED_MODEL_PATH):
        if os.path.isfile(cfg.TRAIN.BACKBONE_PRETRAINED_PATH):
            logging.info('Load backbone pretrained model from {}'.format(
                cfg.TRAIN.BACKBONE_PRETRAINED_PATH
            ))
            msg = model.load_state_dict(torch.load(cfg.TRAIN.BACKBONE_PRETRAINED_PATH))
            logging.info(msg)
        else:
            logging.info('load backbone pretrained model from url..')
            msg = model.load_state_dict(model_zoo.load_url(model_urls[backbone]))
            logging.info(msg)
    return model


