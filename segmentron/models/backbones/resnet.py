import os
import logging
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo

from ...utils.config import cfg

__all__ = ['ResNetV1', 'get_resnet']

model_urls = {
    'resnet18b': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34b': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50b': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101b': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152b': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnet50c': 'https://github.com/LikeLy-Journey/SegmenTron/releases/download/v0.1.0/resnet50-25c4b509.pth',
    'resnet101c': 'https://github.com/LikeLy-Journey/SegmenTron/releases/download/v0.1.0/resnet101-2a57e44d.pth',
    'resnet152c': 'https://github.com/LikeLy-Journey/SegmenTron/releases/download/v0.1.0/resnet152-0d43d698.pth',
}


class BasicBlockV1b(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None,
                 previous_dilation=1, norm_layer=nn.BatchNorm2d):
        super(BasicBlockV1b, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 3, stride,
                               dilation, dilation, bias=False)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(True)
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, previous_dilation,
                               dilation=previous_dilation, bias=False)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class BottleneckV1b(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None,
                 previous_dilation=1, norm_layer=nn.BatchNorm2d):
        super(BottleneckV1b, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = norm_layer(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride,
                               dilation, dilation, bias=False)
        self.bn2 = norm_layer(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNetV1(nn.Module):

    def __init__(self, block, layers, num_classes=1000, dilated=True, deep_stem=False,
                 zero_init_residual=False, norm_layer=nn.BatchNorm2d):
        self.inplanes = 128 if deep_stem else 64
        super(ResNetV1, self).__init__()
        if deep_stem:
            self.conv1 = nn.Sequential(
                nn.Conv2d(3, 64, 3, 2, 1, bias=False),
                norm_layer(64),
                nn.ReLU(True),
                nn.Conv2d(64, 64, 3, 1, 1, bias=False),
                norm_layer(64),
                nn.ReLU(True),
                nn.Conv2d(64, 128, 3, 1, 1, bias=False)
            )
        else:
            self.conv1 = nn.Conv2d(3, 64, 7, 2, 3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(True)
        self.maxpool = nn.MaxPool2d(3, 2, 1)
        self.layer1 = self._make_layer(block, 64, layers[0], norm_layer=norm_layer)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, norm_layer=norm_layer)
        if dilated:
            self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation=2, norm_layer=norm_layer)
            self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=4, norm_layer=norm_layer)
        else:
            self.layer3 = self._make_layer(block, 256, layers[2], stride=2, norm_layer=norm_layer)
            self.layer4 = self._make_layer(block, 512, layers[3], stride=2, norm_layer=norm_layer)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BottleneckV1b):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlockV1b):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, norm_layer=nn.BatchNorm2d):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, 1, stride, bias=False),
                norm_layer(planes * block.expansion),
            )

        layers = []
        if dilation in (1, 2):
            layers.append(block(self.inplanes, planes, stride, dilation=1, downsample=downsample,
                                previous_dilation=dilation, norm_layer=norm_layer))
        elif dilation == 4:
            layers.append(block(self.inplanes, planes, stride, dilation=2, downsample=downsample,
                                previous_dilation=dilation, norm_layer=norm_layer))
        else:
            raise RuntimeError("=> unknown dilation size: {}".format(dilation))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation,
                                previous_dilation=dilation, norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        c1 = self.layer1(x)
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)
        c4 = self.layer4(c3)

        # for classification
        # x = self.avgpool(c4)
        # x = x.view(x.size(0), -1)
        # x = self.fc(x)

        return c1, c3, c4


def get_resnet(backbone):
    resnet_blocks = {
        'resnet18': [2, 2, 2, 2],
        'resnet34': [3, 4, 6, 3],
        'resnet50': [3, 4, 6, 3],
        'resnet101': [3, 4, 23, 3],
        'resnet152': [3, 8, 36, 3],
    }
    deep_stem = cfg.MODEL.RESNET_VARIANT == 'c'
    model = ResNetV1(BottleneckV1b, resnet_blocks[backbone], deep_stem=deep_stem)
    if cfg.TRAIN.BACKBONE_PRETRAINED and (not cfg.TRAIN.PRETRAINED_MODEL_PATH):
        if os.path.isfile(cfg.TRAIN.BACKBONE_PRETRAINED_PATH) and os.path.exists(cfg.TRAIN.BACKBONE_PRETRAINED_PATH):
            logging.info('Load backbone pretrained model from {}'.format(
                cfg.TRAIN.BACKBONE_PRETRAINED_PATH
            ))
            msg = model.load_state_dict(torch.load(cfg.TRAIN.BACKBONE_PRETRAINED_PATH))
            logging.info(msg)
        else:
            logging.info('load backbone pretrained model..')
            msg = model.load_state_dict(model_zoo.load_url(model_urls[backbone + cfg.MODEL.RESNET_VARIANT]))
            logging.info(msg)
    return model

