"""Fast Segmentation Convolutional Neural Network"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from .model_zoo import MODEL_REGISTRY
from .segbase import SegBaseModel
from ..modules.basic import SeparableConv2d, _ConvBNReLU, InvertedResidual
from ..modules import PyramidPooling, get_norm
from ..config import cfg

__all__ = ['FastSCNN']


@MODEL_REGISTRY.register()
class FastSCNN(SegBaseModel):
    def __init__(self):
        super(FastSCNN, self).__init__(need_backbone=False)
        self.aux = cfg.SOLVER.AUX
        self.norm_layer = get_norm(cfg.MODEL.BN_TYPE)
        self.learning_to_downsample = LearningToDownsample(32, 48, 64, norm_layer=self.norm_layer)
        self.global_feature_extractor = GlobalFeatureExtractor(64, [64, 96, 128], 128, 6, [3, 3, 3],
                                                               norm_layer=self.norm_layer)
        self.feature_fusion = FeatureFusionModule(64, 128, 128, norm_layer=self.norm_layer)
        self.classifier = Classifier(128, self.nclass, norm_layer=self.norm_layer)

        decoder_list = ['learning_to_downsample', 'global_feature_extractor',
                        'feature_fusion', 'classifier']

        if self.aux:
            self.auxlayer1 = nn.Sequential(
                nn.Conv2d(64, 32, 3, padding=1, bias=False),
                self.norm_layer(32),
                nn.ReLU(True),
                nn.Dropout2d(0.1),
                nn.Conv2d(32, self.nclass, 1)
            )
            self.auxlayer2 = nn.Sequential(
                nn.Conv2d(128, 32, 3, padding=1, bias=False),
                self.norm_layer(32),
                nn.ReLU(True),
                nn.Dropout2d(0.1),
                nn.Conv2d(32, self.nclass, 1)
            )
            decoder_list += ['auxlayer1', 'auxlayer2']

        self.__setattr__('decoder', decoder_list)

    def forward(self, x):
        size = x.size()[2:]
        higher_res_features = self.learning_to_downsample(x)
        lower_res_features = self.global_feature_extractor(higher_res_features)
        x = self.feature_fusion(higher_res_features, lower_res_features)
        x = self.classifier(x)
        outputs = []
        x = F.interpolate(x, size, mode='bilinear', align_corners=True)
        outputs.append(x)
        if self.aux:
            auxout1 = self.auxlayer1(higher_res_features)
            auxout1 = F.interpolate(auxout1, size, mode='bilinear', align_corners=True)
            auxout2 = self.auxlayer2(lower_res_features)
            auxout2 = F.interpolate(auxout2, size, mode='bilinear', align_corners=True)
            outputs.append(auxout1)
            outputs.append(auxout2)
        return {"inference_results": x, "loss_results": tuple(outputs)}


class LearningToDownsample(nn.Module):
    """Learning to downsample module"""

    def __init__(self, dw_channels1=32, dw_channels2=48, out_channels=64, norm_layer=nn.BatchNorm2d):
        super(LearningToDownsample, self).__init__()
        self.conv = _ConvBNReLU(3, dw_channels1, 3, 2)
        self.dsconv1 = SeparableConv2d(dw_channels1, dw_channels2, stride=2, relu_first=False, norm_layer=norm_layer)
        self.dsconv2 = SeparableConv2d(dw_channels2, out_channels, stride=2, relu_first=False, norm_layer=norm_layer)

    def forward(self, x):
        x = self.conv(x)
        x = self.dsconv1(x)
        x = self.dsconv2(x)
        return x


class GlobalFeatureExtractor(nn.Module):
    """Global feature extractor module"""

    def __init__(self, in_channels=64, block_channels=(64, 96, 128), out_channels=128,
                 t=6, num_blocks=(3, 3, 3), norm_layer=nn.BatchNorm2d):
        super(GlobalFeatureExtractor, self).__init__()
        self.bottleneck1 = self._make_layer(InvertedResidual, in_channels, block_channels[0], num_blocks[0],
                                            t, 2, norm_layer=norm_layer)
        self.bottleneck2 = self._make_layer(InvertedResidual, block_channels[0], block_channels[1],
                                            num_blocks[1], t, 2, norm_layer=norm_layer)
        self.bottleneck3 = self._make_layer(InvertedResidual, block_channels[1], block_channels[2],
                                            num_blocks[2], t, 1, norm_layer=norm_layer)
        self.ppm = PyramidPooling(block_channels[2], norm_layer=norm_layer)
        self.out = _ConvBNReLU(block_channels[2] * 2, out_channels, 1, norm_layer=norm_layer)

    def _make_layer(self, block, inplanes, planes, blocks, t=6, stride=1, norm_layer=nn.BatchNorm2d):
        layers = []
        layers.append(block(inplanes, planes, stride, t, norm_layer=norm_layer))
        for i in range(1, blocks):
            layers.append(block(planes, planes, 1, t, norm_layer=norm_layer))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.bottleneck1(x)
        x = self.bottleneck2(x)
        x = self.bottleneck3(x)
        x = self.ppm(x)
        x = self.out(x)
        return x


class FeatureFusionModule(nn.Module):
    """Feature fusion module"""

    def __init__(self, highter_in_channels, lower_in_channels, out_channels, scale_factor=4, norm_layer=nn.BatchNorm2d):
        super(FeatureFusionModule, self).__init__()
        self.scale_factor = scale_factor
        self.dwconv = _ConvBNReLU(lower_in_channels, out_channels, 1, norm_layer=norm_layer)
        self.conv_lower_res = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 1),
            norm_layer(out_channels)
        )
        self.conv_higher_res = nn.Sequential(
            nn.Conv2d(highter_in_channels, out_channels, 1),
            norm_layer(out_channels)
        )
        self.relu = nn.ReLU(True)

    def forward(self, higher_res_feature, lower_res_feature):
        lower_res_feature = F.interpolate(lower_res_feature, scale_factor=4, mode='bilinear', align_corners=True)
        lower_res_feature = self.dwconv(lower_res_feature)
        lower_res_feature = self.conv_lower_res(lower_res_feature)

        higher_res_feature = self.conv_higher_res(higher_res_feature)
        out = higher_res_feature + lower_res_feature
        return self.relu(out)


class Classifier(nn.Module):
    """Classifier"""

    def __init__(self, dw_channels, num_classes, stride=1, norm_layer=nn.BatchNorm2d):
        super(Classifier, self).__init__()
        self.dsconv1 = SeparableConv2d(dw_channels, dw_channels, stride=stride, relu_first=False,
                                       norm_layer=norm_layer)
        self.dsconv2 = SeparableConv2d(dw_channels, dw_channels, stride=stride, relu_first=False,
                                       norm_layer=norm_layer)
        self.conv = nn.Sequential(
            nn.Dropout2d(0.1),
            nn.Conv2d(dw_channels, num_classes, 1)
        )

    def forward(self, x):
        x = self.dsconv1(x)
        x = self.dsconv2(x)
        x = self.conv(x)
        return x
