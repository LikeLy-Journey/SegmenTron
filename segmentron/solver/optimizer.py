import logging
import torch.nn as nn

from torch import optim
from segmentron.config import cfg


def _set_batch_norm_attr(named_modules, attr, value):
    for m in named_modules:
        if isinstance(m[1], (nn.BatchNorm2d, nn.SyncBatchNorm)):
            setattr(m[1], attr, value)


def _get_paramters(model):
    params_list = list()
    if hasattr(model, 'encoder') and model.encoder is not None and hasattr(model, 'decoder'):
        params_list.append({'params': model.encoder.parameters(), 'lr': cfg.SOLVER.LR})
        if cfg.MODEL.BN_EPS_FOR_ENCODER:
            logging.info('Set bn custom eps for bn in encoder: {}'.format(cfg.MODEL.BN_EPS_FOR_ENCODER))
            _set_batch_norm_attr(model.encoder.named_modules(), 'eps', cfg.MODEL.BN_EPS_FOR_ENCODER)

        for module in model.decoder:
            params_list.append({'params': getattr(model, module).parameters(),
                                'lr': cfg.SOLVER.LR * cfg.SOLVER.DECODER_LR_FACTOR})

        if cfg.MODEL.BN_EPS_FOR_DECODER:
            logging.info('Set bn custom eps for bn in decoder: {}'.format(cfg.MODEL.BN_EPS_FOR_DECODER))
            for module in model.decoder:
                _set_batch_norm_attr(getattr(model, module).named_modules(), 'eps',
                                         cfg.MODEL.BN_EPS_FOR_DECODER)
    else:
        logging.info('Model do not have encoder or decoder, params list was from model.parameters(), '
                     'and arguments BN_EPS_FOR_ENCODER, BN_EPS_FOR_DECODER, DECODER_LR_FACTOR not used!')
        params_list = model.parameters()

    if cfg.MODEL.BN_MOMENTUM and cfg.MODEL.BN_TYPE in ['BN']:
        logging.info('Set bn custom momentum: {}'.format(cfg.MODEL.BN_MOMENTUM))
        _set_batch_norm_attr(model.named_modules(), 'momentum', cfg.MODEL.BN_MOMENTUM)
    elif cfg.MODEL.BN_MOMENTUM and cfg.MODEL.BN_TYPE not in ['BN']:
        logging.info('Batch norm type is {}, custom bn momentum is not effective!'.format(cfg.MODEL.BN_TYPE))

    return params_list


def get_optimizer(model):
    parameters = _get_paramters(model)
    opt_lower = cfg.SOLVER.OPTIMIZER.lower()

    if opt_lower == 'sgd':
        optimizer = optim.SGD(
            parameters, lr=cfg.SOLVER.LR, momentum=cfg.SOLVER.MOMENTUM, weight_decay=cfg.SOLVER.WEIGHT_DECAY)
    elif opt_lower == 'adam':
        optimizer = optim.Adam(
            parameters, lr=cfg.SOLVER.LR, eps=cfg.SOLVER.EPSILON, weight_decay=cfg.SOLVER.WEIGHT_DECAY)
    elif opt_lower == 'adadelta':
        optimizer = optim.Adadelta(
            parameters, lr=cfg.SOLVER.LR, eps=cfg.SOLVER.EPSILON, weight_decay=cfg.SOLVER.WEIGHT_DECAY)
    elif opt_lower == 'rmsprop':
        optimizer = optim.RMSprop(
            parameters, lr=cfg.SOLVER.LR, alpha=0.9, eps=cfg.SOLVER.EPSILON,
            momentum=cfg.SOLVER.MOMENTUM, weight_decay=cfg.SOLVER.WEIGHT_DECAY)
    else:
        raise ValueError("Expected optimizer method in [sgd, adam, adadelta, rmsprop], but received "
                         "{}".format(opt_lower))

    return optimizer
