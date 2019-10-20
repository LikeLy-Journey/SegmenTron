from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import codecs
from ast import literal_eval

import yaml
import six
import time

class SegConfig(dict):
    def __init__(self, *args, **kwargs):
        super(SegConfig, self).__init__(*args, **kwargs)
        self.immutable = False

    def __setattr__(self, key, value, create_if_not_exist=True):
        if key in ["immutable"]:
            self.__dict__[key] = value
            return

        t = self
        keylist = key.split(".")
        for k in keylist[:-1]:
            t = t.__getattr__(k, create_if_not_exist)

        t.__getattr__(keylist[-1], create_if_not_exist)
        t[keylist[-1]] = value

    def __getattr__(self, key, create_if_not_exist=True):
        if key in ["immutable"]:
            return self.__dict__[key]

        if not key in self:
            if not create_if_not_exist:
                raise KeyError
            self[key] = SegConfig()
        return self[key]

    def __setitem__(self, key, value):
        #
        if self.immutable:
            raise AttributeError(
                'Attempted to set "{}" to "{}", but SegConfig is immutable'.
                format(key, value))
        #
        if isinstance(value, six.string_types):
            try:
                value = literal_eval(value)
            except ValueError:
                pass
            except SyntaxError:
                pass
        super(SegConfig, self).__setitem__(key, value)

    def update_from_segconfig(self, other):
        if isinstance(other, dict):
            other = SegConfig(other)
        assert isinstance(other, SegConfig)
        diclist = [("", other)]
        while len(diclist):
            prefix, tdic = diclist[0]
            diclist = diclist[1:]
            for key, value in tdic.items():
                key = "{}.{}".format(prefix, key) if prefix else key
                if isinstance(value, dict):
                    diclist.append((key, value))
                    continue
                try:
                    self.__setattr__(key, value, create_if_not_exist=False)
                except KeyError:
                    raise KeyError('Non-existent config key: {}'.format(key))

    def check_and_infer(self, reset_dataset=False):
        cfg.TIME_STAMP = time.strftime('%Y-%m-%d-%H-%M', time.localtime())

    def update_from_list(self, config_list):
        if len(config_list) % 2 != 0:
            raise ValueError(
                "Command line options config format error! Please check it: {}".
                format(config_list))
        for key, value in zip(config_list[0::2], config_list[1::2]):
            try:
                self.__setattr__(key, value, create_if_not_exist=False)
            except KeyError:
                raise KeyError('Non-existent config key: {}'.format(key))

    def update_from_file(self, config_file):
        with codecs.open(config_file, 'r', 'utf-8') as file:
            dic = yaml.load(file, Loader=yaml.FullLoader)
        self.update_from_segconfig(dic)

    def set_immutable(self, immutable):
        self.immutable = immutable
        for value in self.values():
            if isinstance(value, SegConfig):
                value.set_immutable(immutable)

    def is_immutable(self):
        return self.immutable

cfg = SegConfig()

########################## basic set ###########################################
# mean
cfg.MEAN = [0.5, 0.5, 0.5]
# std
cfg.STD = [0.5, 0.5, 0.5]
# batch size
cfg.BATCH_SIZE = 1
# eval crop size
cfg.EVAL_CROP_SIZE = tuple()
# train crop size
cfg.TRAIN_CROP_SIZE = 769
# train base size
cfg.TRAIN_BASE_SIZE = 1024
# epochs
cfg.EPOCHS = 30
# train time stamp
cfg.TIME_STAMP = ''

########################## dataset config #########################################
# dataset name
cfg.DATASET = ''
# dataset ignore index
cfg.IGNORE_INDEX = -1
# workers
cfg.WORKERS = 4

########################### data augment ######################################
# whether to use image blur
cfg.AUG.RICH_CROP.BLUR = False
# data augment blur rator
cfg.AUG.RICH_CROP.BLUR_RATIO = 0.1

########################### 训练配置 ##########################################
# model output dir
cfg.TRAIN.MODEL_SAVE_DIR = '../runs/checkpoints/'
# log dir
cfg.TRAIN.LOG_SAVE_DIR = '../runs/logs/'
# pretrained model for eval or finetune
cfg.TRAIN.PRETRAINED_MODEL_PATH = ''
#
cfg.TRAIN.BACKBONE_PRETRAINED = True
# backbone pretrained model path, if not specific, will load from url when backbone pretrained enabled
cfg.TRAIN.BACKBONE_PRETRAINED_PATH = ''
# resume model path
cfg.TRAIN.RESUME_MODEL_PATH = ''
# whether to use synchronize bn
cfg.TRAIN.SYNC_BATCH_NORM = True
# save model every checkpoint-epoch
cfg.TRAIN.SNAPSHOT_EPOCH = 5

########################### 模型优化相关配置 ##################################
# base learning rate
cfg.SOLVER.LR = 1e-4
#
cfg.SOLVER.DECODER_LR_FACTOR = 10.0
# lr scheduler mode
cfg.SOLVER.LR_SCHEDULER = "poly"
# optimizer mother
cfg.SOLVER.OPTIMIZER = "sgd"
# optimizer momentum
cfg.SOLVER.MOMENTUM = 0.9
# momentum2
cfg.SOLVER.MOMENTUM2 = 0.999
# poly power
cfg.SOLVER.POWER = 0.9
# step gamma
cfg.SOLVER.GAMMA = 0.1
#
cfg.SOLVER.EPSILON = 1e-5
# step milestone
cfg.SOLVER.DECAY_EPOCH = [10, 20]
# weight decay
cfg.SOLVER.WEIGHT_DECAY = 1e-4 #0.00004
#
cfg.SOLVER.WARMUP_ITERS = 0
#
cfg.SOLVER.WARMUP_FACTOR = 1.0 / 3
#
cfg.SOLVER.WARMUP_METHOD = 'linear'
# start epoch
cfg.SOLVER.BEGIN_EPOCH = 1
#
cfg.SOLVER.START_EPOCH = 0
# loss softmax_loss, bce_loss, dice_loss
cfg.SOLVER.LOSS = ["softmax_loss"]
#
cfg.SOLVER.OHEM = False
#
cfg.SOLVER.AUX = False
#
cfg.SOLVER.AUX_WEIGHT = 0.4

########################## 测试配置 ###########################################
# 测试模型路径
cfg.TEST.TEST_MODEL = ''

########################## 模型通用配置 #######################################
# model name
cfg.MODEL.MODEL_NAME = ''
# model backbone
cfg.MODEL.BACKBONE = ''
# support resnet b, c. b is standard resnet in pytorch official repo
cfg.MODEL.RESNET_VARIANT = 'b'
# BatchNorm: bn、gn(group_norm)
cfg.MODEL.DEFAULT_NORM_TYPE = 'bn'
#
cfg.MODEL.MULTI_LOSS_WEIGHT = [1.0]
# gn groups
cfg.MODEL.DEFAULT_GROUP_NUMBER = 32
#
cfg.MODEL.DEFAULT_EPSILON = 1e-5
#
cfg.MODEL.BN_EPS_FOR_ENCODER = 1e-3
#
cfg.MODEL.BN_EPS_FOR_DECODER = None
#
cfg.MODEL.OUTPUT_STRIDE = 16
# BatchNorm momentum
cfg.MODEL.BN_MOMENTUM = None


########################## DeepLab config ####################################
# DeepLab backbone
cfg.MODEL.DEEPLAB.BACKBONE = "xception_65"
# DeepLab output stride
cfg.MODEL.DEEPLAB.OUTPUT_STRIDE = 16
# MobileNet backbone scale
cfg.MODEL.DEEPLAB.DEPTH_MULTIPLIER = 1.0
# MobileNet backbone scale
cfg.MODEL.DEEPLAB.ENCODER_WITH_ASPP = True
# MobileNet backbone scale
cfg.MODEL.DEEPLAB.ENABLE_DECODER = True
# ASPP whether use sep conv
cfg.MODEL.DEEPLAB.ASPP_WITH_SEP_CONV = True
# decoder whether use sep conv
cfg.MODEL.DEEPLAB.DECODER_USE_SEP_CONV = True

########################## UNET config #######################################
# upsample mode
cfg.MODEL.UNET.UPSAMPLE_MODE = 'bilinear'

########################## ICNET config ######################################
# RESNET backbone scale
cfg.MODEL.ICNET.DEPTH_MULTIPLIER = 0.5
# RESNET layers
cfg.MODEL.ICNET.LAYERS = 50

########################## PSPNET config ######################################
# RESNET backbone scale
cfg.MODEL.PSPNET.DEPTH_MULTIPLIER = 1
# RESNET backbone
cfg.MODEL.PSPNET.LAYERS = 50



