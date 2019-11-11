import torch
import logging

from .deeplabv3_plus import get_deeplabv3_plus
from .danet import get_danet
from .icnet import get_icnet
from .pspnet import get_psp
from .dfanet import get_dfanet
from .fast_scnn import get_fast_scnn
from .fcn import get_fcn
from .hrnet_seg import get_hrnet
from ..config import cfg

__all__ = ['get_model_list', 'get_segmentation_model']

_models = {
        'deeplabv3_plus': get_deeplabv3_plus,
        'danet': get_danet,
        'pspnet': get_psp,
        'icnet': get_icnet,
        'dfanet': get_dfanet,
        'fast_scnn': get_fast_scnn,
        'fcn': get_fcn,
        'hrnet': get_hrnet,
    }

def get_model_list():
    return list(_models.keys())


def get_segmentation_model():
    model = _models[cfg.MODEL.MODEL_NAME]()
    if cfg.PHASE == 'train':
        if cfg.TRAIN.PRETRAINED_MODEL_PATH:
            logging.info('load pretrained model from {}'.format(cfg.TRAIN.PRETRAINED_MODEL_PATH))
            msg = model.load_state_dict(torch.load(cfg.TRAIN.PRETRAINED_MODEL_PATH))
            logging.info(msg)
    else:
        if cfg.TEST.TEST_MODEL_PATH:
            logging.info('load test model from {}'.format(cfg.TEST.TEST_MODEL_PATH))
            msg = model.load_state_dict(torch.load(cfg.TEST.TEST_MODEL_PATH))
            logging.info(msg)
    return model
