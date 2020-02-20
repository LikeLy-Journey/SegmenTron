import torch.nn as nn

from ...modules import SeparableConv2d
from .build import BACKBONE_REGISTRY
from ...config import cfg

__all__ = ['Xception65', 'Enc', 'FCAttention']


class XceptionBlock(nn.Module):
    def __init__(self, channel_list, stride=1, dilation=1, skip_connection_type='conv', relu_first=True,
                 low_feat=False, norm_layer=nn.BatchNorm2d):
        super().__init__()

        assert len(channel_list) == 4
        self.skip_connection_type = skip_connection_type
        self.relu_first = relu_first
        self.low_feat = low_feat

        if self.skip_connection_type == 'conv':
            self.conv = nn.Conv2d(channel_list[0], channel_list[-1], 1, stride=stride, bias=False)
            self.bn = norm_layer(channel_list[-1])

        self.sep_conv1 = SeparableConv2d(channel_list[0], channel_list[1], dilation=dilation,
                                         relu_first=relu_first, norm_layer=norm_layer)
        self.sep_conv2 = SeparableConv2d(channel_list[1], channel_list[2], dilation=dilation,
                                         relu_first=relu_first, norm_layer=norm_layer)
        self.sep_conv3 = SeparableConv2d(channel_list[2], channel_list[3], dilation=dilation,
                                         relu_first=relu_first, stride=stride, norm_layer=norm_layer)
        self.last_inp_channels = channel_list[3]

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
    def __init__(self, norm_layer=nn.BatchNorm2d):
        super().__init__()
        output_stride = cfg.MODEL.OUTPUT_STRIDE
        if output_stride == 32:
            entry_block3_stride = 2
            middle_block_dilation = 1
            exit_block_dilations = (1, 1)
            exit_block_stride = 2
        elif output_stride == 16:
            entry_block3_stride = 2
            middle_block_dilation = 1
            exit_block_dilations = (1, 2)
            exit_block_stride = 1
        elif output_stride == 8:
            entry_block3_stride = 1
            middle_block_dilation = 2
            exit_block_dilations = (2, 4)
            exit_block_stride = 1
        else:
            raise NotImplementedError

        # Entry flow
        self.conv1 = nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False)
        self.bn1 = norm_layer(32)
        self.relu = nn.ReLU()

        self.conv2 = nn.Conv2d(32, 64, 3, stride=1, padding=1, bias=False)
        self.bn2 = norm_layer(64)

        self.block1 = XceptionBlock([64, 128, 128, 128], stride=2, norm_layer=norm_layer)
        self.block2 = XceptionBlock([128, 256, 256, 256], stride=2, low_feat=True, norm_layer=norm_layer)
        self.block3 = XceptionBlock([256, 728, 728, 728], stride=entry_block3_stride, low_feat=True,
                                    norm_layer=norm_layer)

        # Middle flow (16 units)
        self.block4 = XceptionBlock([728, 728, 728, 728], dilation=middle_block_dilation,
                                    skip_connection_type='sum', norm_layer=norm_layer)
        self.block5 = XceptionBlock([728, 728, 728, 728], dilation=middle_block_dilation,
                                    skip_connection_type='sum', norm_layer=norm_layer)
        self.block6 = XceptionBlock([728, 728, 728, 728], dilation=middle_block_dilation,
                                    skip_connection_type='sum', norm_layer=norm_layer)
        self.block7 = XceptionBlock([728, 728, 728, 728], dilation=middle_block_dilation,
                                    skip_connection_type='sum', norm_layer=norm_layer)
        self.block8 = XceptionBlock([728, 728, 728, 728], dilation=middle_block_dilation,
                                    skip_connection_type='sum', norm_layer=norm_layer)
        self.block9 = XceptionBlock([728, 728, 728, 728], dilation=middle_block_dilation,
                                    skip_connection_type='sum', norm_layer=norm_layer)
        self.block10 = XceptionBlock([728, 728, 728, 728], dilation=middle_block_dilation,
                                     skip_connection_type='sum', norm_layer=norm_layer)
        self.block11 = XceptionBlock([728, 728, 728, 728], dilation=middle_block_dilation,
                                     skip_connection_type='sum', norm_layer=norm_layer)
        self.block12 = XceptionBlock([728, 728, 728, 728], dilation=middle_block_dilation,
                                     skip_connection_type='sum', norm_layer=norm_layer)
        self.block13 = XceptionBlock([728, 728, 728, 728], dilation=middle_block_dilation,
                                     skip_connection_type='sum', norm_layer=norm_layer)
        self.block14 = XceptionBlock([728, 728, 728, 728], dilation=middle_block_dilation,
                                     skip_connection_type='sum', norm_layer=norm_layer)
        self.block15 = XceptionBlock([728, 728, 728, 728], dilation=middle_block_dilation,
                                     skip_connection_type='sum', norm_layer=norm_layer)
        self.block16 = XceptionBlock([728, 728, 728, 728], dilation=middle_block_dilation,
                                     skip_connection_type='sum', norm_layer=norm_layer)
        self.block17 = XceptionBlock([728, 728, 728, 728], dilation=middle_block_dilation,
                                     skip_connection_type='sum', norm_layer=norm_layer)
        self.block18 = XceptionBlock([728, 728, 728, 728], dilation=middle_block_dilation,
                                     skip_connection_type='sum', norm_layer=norm_layer)
        self.block19 = XceptionBlock([728, 728, 728, 728], dilation=middle_block_dilation,
                                     skip_connection_type='sum', norm_layer=norm_layer)

        # Exit flow
        self.block20 = XceptionBlock([728, 728, 1024, 1024], stride=exit_block_stride,
                                     dilation=exit_block_dilations[0], norm_layer=norm_layer)
        self.block21 = XceptionBlock([1024, 1536, 1536, 2048], dilation=exit_block_dilations[1],
                                     skip_connection_type='none', relu_first=False, norm_layer=norm_layer)

    def forward(self, x):
        # Entry flow
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.block1(x)
        x, c1 = self.block2(x)  # b, h//4, w//4, 256
        x, c2 = self.block3(x)  # b, h//8, w//8, 728

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
        c3 = self.block19(x)

        # Exit flow
        x = self.block20(c3)
        c4 = self.block21(x)

        return c1, c2, c3, c4


# -------------------------------------------------
#                   For DFANet
# -------------------------------------------------
class BlockA(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, dilation=1, norm_layer=None, start_with_relu=True):
        super(BlockA, self).__init__()
        if out_channels != in_channels or stride != 1:
            self.skip = nn.Conv2d(in_channels, out_channels, 1, stride, bias=False)
            self.skipbn = norm_layer(out_channels)
        else:
            self.skip = None
        self.relu = nn.ReLU()
        rep = list()
        inter_channels = out_channels // 4

        if start_with_relu:
            rep.append(self.relu)
        rep.append(SeparableConv2d(in_channels, inter_channels, 3, 1, dilation, norm_layer=norm_layer))
        rep.append(norm_layer(inter_channels))

        rep.append(self.relu)
        rep.append(SeparableConv2d(inter_channels, inter_channels, 3, 1, dilation, norm_layer=norm_layer))
        rep.append(norm_layer(inter_channels))

        if stride != 1:
            rep.append(self.relu)
            rep.append(SeparableConv2d(inter_channels, out_channels, 3, stride, norm_layer=norm_layer))
            rep.append(norm_layer(out_channels))
        else:
            rep.append(self.relu)
            rep.append(SeparableConv2d(inter_channels, out_channels, 3, 1, norm_layer=norm_layer))
            rep.append(norm_layer(out_channels))
        self.rep = nn.Sequential(*rep)

    def forward(self, x):
        out = self.rep(x)
        if self.skip is not None:
            skip = self.skipbn(self.skip(x))
        else:
            skip = x
        out = out + skip
        return out


class Enc(nn.Module):
    def __init__(self, in_channels, out_channels, blocks, norm_layer=nn.BatchNorm2d):
        super(Enc, self).__init__()
        block = list()
        block.append(BlockA(in_channels, out_channels, 2, norm_layer=norm_layer))
        for i in range(blocks - 1):
            block.append(BlockA(out_channels, out_channels, 1, norm_layer=norm_layer))
        self.block = nn.Sequential(*block)

    def forward(self, x):
        return self.block(x)


class FCAttention(nn.Module):
    def __init__(self, in_channels, norm_layer=nn.BatchNorm2d):
        super(FCAttention, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(in_channels, 1000)
        self.conv = nn.Sequential(
            nn.Conv2d(1000, in_channels, 1, bias=False),
            norm_layer(in_channels),
            nn.ReLU(True))

    def forward(self, x):
        n, c, _, _ = x.size()
        att = self.avgpool(x).view(n, c)
        att = self.fc(att).view(n, 1000, 1, 1)
        att = self.conv(att)
        return x * att.expand_as(x)


class XceptionA(nn.Module):
    def __init__(self, num_classes=1000, norm_layer=nn.BatchNorm2d):
        super(XceptionA, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(3, 8, 3, 2, 1, bias=False),
                                   norm_layer(8),
                                   nn.ReLU(True))

        self.enc2 = Enc(8, 48, 4, norm_layer=norm_layer)
        self.enc3 = Enc(48, 96, 6, norm_layer=norm_layer)
        self.enc4 = Enc(96, 192, 4, norm_layer=norm_layer)

        self.fca = FCAttention(192, norm_layer=norm_layer)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(192, num_classes)

    def forward(self, x):
        x = self.conv1(x)

        x = self.enc2(x)
        x = self.enc3(x)
        x = self.enc4(x)
        x = self.fca(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


@BACKBONE_REGISTRY.register()
def xception_a(norm_layer=nn.BatchNorm2d):
    model = XceptionA(norm_layer=norm_layer)
    return model


@BACKBONE_REGISTRY.register()
def xception65(norm_layer=nn.BatchNorm2d):
    model = Xception65(norm_layer=norm_layer)
    return model

