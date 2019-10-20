import torch
import logging
from .deeplabv3_plus import *
from ..utils.config import cfg
__all__ = ['get_model', 'get_model_list', 'get_segmentation_model']

_models = {
    'deeplabv3_plus_xception_voc': get_deeplabv3_plus_xception_voc,
}


def get_model(name, **kwargs):
    name = name.lower()
    if name not in _models:
        err_str = '"%s" is not among the following model list:\n\t' % (name)
        err_str += '%s' % ('\n\t'.join(sorted(_models.keys())))
        raise ValueError(err_str)
    net = _models[name](**kwargs)
    return net


def get_model_list():
    return _models.keys()


def get_segmentation_model():
    models = {
        'deeplabv3_plus': get_deeplabv3_plus,
    }
    model = models[cfg.MODEL.MODEL_NAME]()
    if cfg.TRAIN.PRETRAINED_MODEL_PATH:
        logging.info('load pretrained model from {}'.format(cfg.TRAIN.PRETRAINED_MODEL_PATH))
        msg = model.load_state_dict(torch.load(cfg.TRAIN.PRETRAINED_MODEL_PATH))
        logging.info(msg)
    return model
