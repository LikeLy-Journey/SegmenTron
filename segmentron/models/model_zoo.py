import logging
import torch

from segmentron.utils.registry import Registry
from ..config import cfg

MODEL_REGISTRY = Registry("MODEL")
MODEL_REGISTRY.__doc__ = """
Registry for segment model, i.e. the whole model.

The registered object will be called with `obj()`
and expected to return a `nn.Module` object.
"""


def get_segmentation_model():
    """
    Built the whole model, defined by `cfg.MODEL.META_ARCHITECTURE`.
    """
    model_name = cfg.MODEL.MODEL_NAME
    model = MODEL_REGISTRY.get(model_name)()
    load_model_pretrain(model)
    return model


def load_model_pretrain(model):
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