import os
import torch
import logging
import torch.utils.model_zoo as model_zoo

from ...config import cfg


model_urls = {
    'resnet18b': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34b': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50b': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101b': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152b': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnet50c': 'https://github.com/LikeLy-Journey/SegmenTron/releases/download/v0.1.0/resnet50-25c4b509.pth',
    'resnet101c': 'https://github.com/LikeLy-Journey/SegmenTron/releases/download/v0.1.0/resnet101-2a57e44d.pth',
    'resnet152c': 'https://github.com/LikeLy-Journey/SegmenTron/releases/download/v0.1.0/resnet152-0d43d698.pth',
    'xception65': 'https://github.com/LikeLy-Journey/SegmenTron/releases/download/v0.1.0/tf-xception65-270e81cf.pth'
}


def load_backbone_pretrained(model, backbone):
    if cfg.PHASE == 'train' and cfg.TRAIN.BACKBONE_PRETRAINED and (not cfg.TRAIN.PRETRAINED_MODEL_PATH):
        if os.path.isfile(cfg.TRAIN.BACKBONE_PRETRAINED_PATH):
            logging.info('Load backbone pretrained model from {}'.format(
                cfg.TRAIN.BACKBONE_PRETRAINED_PATH
            ))
            msg = model.load_state_dict(torch.load(cfg.TRAIN.BACKBONE_PRETRAINED_PATH), strict=False)
            logging.info(msg)
        elif backbone not in model_urls:
            logging.info('{} has no pretrained model'.format(backbone))
            return
        else:
            logging.info('load backbone pretrained model from url..')
            msg = model.load_state_dict(model_zoo.load_url(model_urls[backbone]))
            logging.info(msg)