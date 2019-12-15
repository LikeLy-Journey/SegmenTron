"""MobileNet and MobileNetV2."""
import torch.nn as nn

from .build import BACKBONE_REGISTRY
from ...modules import _ConvBNReLU, _DepthwiseConv, InvertedResidual
from ...config import cfg

__all__ = ['MobileNet', 'MobileNetV2']


class MobileNet(nn.Module):
    def __init__(self, num_classes=1000, norm_layer=nn.BatchNorm2d):
        super(MobileNet, self).__init__()
        multiplier = cfg.MODEL.BACKBONE_SCALE
        conv_dw_setting = [
            [64, 1, 1],
            [128, 2, 2],
            [256, 2, 2],
            [512, 6, 2],
            [1024, 2, 2]]
        input_channels = int(32 * multiplier) if multiplier > 1.0 else 32
        features = [_ConvBNReLU(3, input_channels, 3, 2, 1, norm_layer=norm_layer)]

        for c, n, s in conv_dw_setting:
            out_channels = int(c * multiplier)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(_DepthwiseConv(input_channels, out_channels, stride, norm_layer))
                input_channels = out_channels
        self.last_inp_channels = int(1024 * multiplier)
        features.append(nn.AdaptiveAvgPool2d(1))
        self.features = nn.Sequential(*features)

        self.classifier = nn.Linear(int(1024 * multiplier), num_classes)

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x.view(x.size(0), x.size(1)))
        return x


class MobileNetV2(nn.Module):
    def __init__(self, num_classes=1000, norm_layer=nn.BatchNorm2d):
        super(MobileNetV2, self).__init__()
        output_stride = cfg.MODEL.OUTPUT_STRIDE
        self.multiplier = cfg.MODEL.BACKBONE_SCALE
        if output_stride == 32:
            dilations = [1, 1]
        elif output_stride == 16:
            dilations = [1, 2]
        elif output_stride == 8:
            dilations = [2, 4]
        else:
            raise NotImplementedError
        inverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1]]
        # building first layer
        input_channels = int(32 * self.multiplier) if self.multiplier > 1.0 else 32
        # last_channels = int(1280 * multiplier) if multiplier > 1.0 else 1280
        self.conv1 = _ConvBNReLU(3, input_channels, 3, 2, 1, relu6=True, norm_layer=norm_layer)

        # building inverted residual blocks
        self.planes = input_channels
        self.block1 = self._make_layer(InvertedResidual, self.planes, inverted_residual_setting[0:1],
                                       norm_layer=norm_layer)
        self.block2 = self._make_layer(InvertedResidual, self.planes, inverted_residual_setting[1:2],
                                       norm_layer=norm_layer)
        self.block3 = self._make_layer(InvertedResidual, self.planes, inverted_residual_setting[2:3],
                                       norm_layer=norm_layer)
        self.block4 = self._make_layer(InvertedResidual, self.planes, inverted_residual_setting[3:5],
                                       dilations[0], norm_layer=norm_layer)
        self.block5 = self._make_layer(InvertedResidual, self.planes, inverted_residual_setting[5:],
                                       dilations[1], norm_layer=norm_layer)
        self.last_inp_channels = self.planes

        # building last several layers
        # features = list()
        # features.append(_ConvBNReLU(input_channels, last_channels, 1, relu6=True, norm_layer=norm_layer))
        # features.append(nn.AdaptiveAvgPool2d(1))
        # self.features = nn.Sequential(*features)
        #
        # self.classifier = nn.Sequential(
        #     nn.Dropout2d(0.2),
        #     nn.Linear(last_channels, num_classes))

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _make_layer(self, block, planes, inverted_residual_setting, dilation=1, norm_layer=nn.BatchNorm2d):
        features = list()
        for t, c, n, s in inverted_residual_setting:
            out_channels = int(c * self.multiplier)
            stride = s if dilation == 1 else 1
            features.append(block(planes, out_channels, stride, t, dilation, norm_layer))
            planes = out_channels
            for i in range(n - 1):
                features.append(block(planes, out_channels, 1, t, norm_layer=norm_layer))
                planes = out_channels
        self.planes = planes
        return nn.Sequential(*features)

    def forward(self, x):
        x = self.conv1(x)
        x = self.block1(x)
        c1 = self.block2(x)
        c2 = self.block3(c1)
        c3 = self.block4(c2)
        c4 = self.block5(c3)

        # x = self.features(x)
        # x = self.classifier(x.view(x.size(0), x.size(1)))
        return c1, c2, c3, c4


@BACKBONE_REGISTRY.register()
def mobilenet_v1(norm_layer=nn.BatchNorm2d):
    return MobileNet(norm_layer=norm_layer)


@BACKBONE_REGISTRY.register()
def mobilenet_v2(norm_layer=nn.BatchNorm2d):
    return MobileNetV2(norm_layer=norm_layer)

