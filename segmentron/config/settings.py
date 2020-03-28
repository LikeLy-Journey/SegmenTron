from .config import SegmentronConfig

cfg = SegmentronConfig()

########################## basic set ###########################################
# random seed
cfg.SEED = 1024
# train time stamp, auto generate, do not need to set
cfg.TIME_STAMP = ''
# root path
cfg.ROOT_PATH = ''
# model phase ['train', 'test']
cfg.PHASE = 'train'

########################## dataset config #########################################
# dataset name
cfg.DATASET.NAME = ''
# pixel mean
cfg.DATASET.MEAN = [0.5, 0.5, 0.5]
# pixel std
cfg.DATASET.STD = [0.5, 0.5, 0.5]
# dataset ignore index
cfg.DATASET.IGNORE_INDEX = -1
# workers
cfg.DATASET.WORKERS = 4
# val dataset mode
cfg.DATASET.MODE = 'testval'
########################### data augment ######################################
# data augment image mirror
cfg.AUG.MIRROR = True
# blur probability
cfg.AUG.BLUR_PROB = 0.0
# blur radius
cfg.AUG.BLUR_RADIUS = 0.0
# color jitter, float or tuple: (0.1, 0.2, 0.3, 0.4)
cfg.AUG.COLOR_JITTER = None
########################### train config ##########################################
# epochs
cfg.TRAIN.EPOCHS = 30
# batch size
cfg.TRAIN.BATCH_SIZE = 1
# train crop size
cfg.TRAIN.CROP_SIZE = 769
# train base size
cfg.TRAIN.BASE_SIZE = 1024
# model output dir
cfg.TRAIN.MODEL_SAVE_DIR = 'runs/checkpoints/'
# log dir
cfg.TRAIN.LOG_SAVE_DIR = 'runs/logs/'
# pretrained model for eval or finetune
cfg.TRAIN.PRETRAINED_MODEL_PATH = ''
# use pretrained backbone model over imagenet
cfg.TRAIN.BACKBONE_PRETRAINED = True
# backbone pretrained model path, if not specific, will load from url when backbone pretrained enabled
cfg.TRAIN.BACKBONE_PRETRAINED_PATH = ''
# resume model path
cfg.TRAIN.RESUME_MODEL_PATH = ''
# whether to use synchronize bn
cfg.TRAIN.SYNC_BATCH_NORM = True
# save model every checkpoint-epoch
cfg.TRAIN.SNAPSHOT_EPOCH = 10

########################### optimizer config ##################################
# base learning rate
cfg.SOLVER.LR = 1e-4
# optimizer method
cfg.SOLVER.OPTIMIZER = "sgd"
# optimizer epsilon
cfg.SOLVER.EPSILON = 1e-8
# optimizer momentum
cfg.SOLVER.MOMENTUM = 0.9
# weight decay
cfg.SOLVER.WEIGHT_DECAY = 1e-4 #0.00004
# decoder lr x10
cfg.SOLVER.DECODER_LR_FACTOR = 10.0
# lr scheduler mode
cfg.SOLVER.LR_SCHEDULER = "poly"
# poly power
cfg.SOLVER.POLY.POWER = 0.9
# step gamma
cfg.SOLVER.STEP.GAMMA = 0.1
# milestone of step lr scheduler
cfg.SOLVER.STEP.DECAY_EPOCH = [10, 20]
# warm up epochs can be float
cfg.SOLVER.WARMUP.EPOCHS = 0.
# warm up factor
cfg.SOLVER.WARMUP.FACTOR = 1.0 / 3
# warm up method
cfg.SOLVER.WARMUP.METHOD = 'linear'
# whether to use ohem
cfg.SOLVER.OHEM = False
# whether to use aux loss
cfg.SOLVER.AUX = False
# aux loss weight
cfg.SOLVER.AUX_WEIGHT = 0.4
# loss name
cfg.SOLVER.LOSS_NAME = ''
########################## test config ###########################################
# val/test model path
cfg.TEST.TEST_MODEL_PATH = ''
# test batch size
cfg.TEST.BATCH_SIZE = 1
# eval crop size
cfg.TEST.CROP_SIZE = None
# multiscale eval
cfg.TEST.SCALES = [1.0]
# flip
cfg.TEST.FLIP = False

########################## visual config ###########################################
# visual result output dir
cfg.VISUAL.OUTPUT_DIR = '../runs/visual/'

########################## model #######################################
# model name
cfg.MODEL.MODEL_NAME = ''
# model backbone
cfg.MODEL.BACKBONE = ''
# model backbone channel scale
cfg.MODEL.BACKBONE_SCALE = 1.0
# support resnet b, c. b is standard resnet in pytorch official repo
# cfg.MODEL.RESNET_VARIANT = 'b'
# multi branch loss weight
cfg.MODEL.MULTI_LOSS_WEIGHT = [1.0]
# gn groups
cfg.MODEL.DEFAULT_GROUP_NUMBER = 32
# whole model default epsilon
cfg.MODEL.DEFAULT_EPSILON = 1e-5
# batch norm, support ['BN', 'SyncBN', 'FrozenBN', 'GN', 'nnSyncBN']
cfg.MODEL.BN_TYPE = 'BN'
# batch norm epsilon for encoder, if set None will use api default value.
cfg.MODEL.BN_EPS_FOR_ENCODER = None
# batch norm epsilon for encoder, if set None will use api default value.
cfg.MODEL.BN_EPS_FOR_DECODER = None
# backbone output stride
cfg.MODEL.OUTPUT_STRIDE = 16
# BatchNorm momentum, if set None will use api default value.
cfg.MODEL.BN_MOMENTUM = None

########################## DANet config ####################################
# danet param
cfg.MODEL.DANET.MULTI_DILATION = None
# danet param
cfg.MODEL.DANET.MULTI_GRID = False

########################## DeepLab config ####################################
# whether to use aspp
cfg.MODEL.DEEPLABV3_PLUS.USE_ASPP = True
# whether to use decoder
cfg.MODEL.DEEPLABV3_PLUS.ENABLE_DECODER = True
# whether aspp use sep conv
cfg.MODEL.DEEPLABV3_PLUS.ASPP_WITH_SEP_CONV = True
# whether decoder use sep conv
cfg.MODEL.DEEPLABV3_PLUS.DECODER_USE_SEP_CONV = True

########################## UNET config #######################################
# upsample mode
# cfg.MODEL.UNET.UPSAMPLE_MODE = 'bilinear'

########################## OCNet config ######################################
# ['base', 'pyramid', 'asp']
cfg.MODEL.OCNet.OC_ARCH = 'base'

########################## EncNet config ######################################
cfg.MODEL.ENCNET.SE_LOSS = True
cfg.MODEL.ENCNET.SE_WEIGHT = 0.2
cfg.MODEL.ENCNET.LATERAL = True


########################## CCNET config ######################################
cfg.MODEL.CCNET.RECURRENCE = 2

########################## CGNET config ######################################
cfg.MODEL.CGNET.STAGE2_BLOCK_NUM = 3
cfg.MODEL.CGNET.STAGE3_BLOCK_NUM = 21

########################## PointRend config ##################################
cfg.MODEL.POINTREND.BASEMODEL = 'DeepLabV3_Plus'

########################## hrnet config ######################################
cfg.MODEL.HRNET.PRETRAINED_LAYERS = ['*']
cfg.MODEL.HRNET.STEM_INPLANES = 64
cfg.MODEL.HRNET.FINAL_CONV_KERNEL = 1
cfg.MODEL.HRNET.WITH_HEAD = True
# stage 1
cfg.MODEL.HRNET.STAGE1.NUM_MODULES = 1
cfg.MODEL.HRNET.STAGE1.NUM_BRANCHES = 1
cfg.MODEL.HRNET.STAGE1.NUM_BLOCKS = [1]
cfg.MODEL.HRNET.STAGE1.NUM_CHANNELS = [32]
cfg.MODEL.HRNET.STAGE1.BLOCK = 'BOTTLENECK'
cfg.MODEL.HRNET.STAGE1.FUSE_METHOD = 'SUM'
# stage 2
cfg.MODEL.HRNET.STAGE2.NUM_MODULES = 1
cfg.MODEL.HRNET.STAGE2.NUM_BRANCHES = 2
cfg.MODEL.HRNET.STAGE2.NUM_BLOCKS = [4, 4]
cfg.MODEL.HRNET.STAGE2.NUM_CHANNELS = [32, 64]
cfg.MODEL.HRNET.STAGE2.BLOCK = 'BASIC'
cfg.MODEL.HRNET.STAGE2.FUSE_METHOD = 'SUM'
# stage 3
cfg.MODEL.HRNET.STAGE3.NUM_MODULES = 1
cfg.MODEL.HRNET.STAGE3.NUM_BRANCHES = 3
cfg.MODEL.HRNET.STAGE3.NUM_BLOCKS = [4, 4, 4]
cfg.MODEL.HRNET.STAGE3.NUM_CHANNELS = [32, 64, 128]
cfg.MODEL.HRNET.STAGE3.BLOCK = 'BASIC'
cfg.MODEL.HRNET.STAGE3.FUSE_METHOD = 'SUM'
# stage 4
cfg.MODEL.HRNET.STAGE4.NUM_MODULES = 1
cfg.MODEL.HRNET.STAGE4.NUM_BRANCHES = 4
cfg.MODEL.HRNET.STAGE4.NUM_BLOCKS = [4, 4, 4, 4]
cfg.MODEL.HRNET.STAGE4.NUM_CHANNELS = [32, 64, 128, 256]
cfg.MODEL.HRNET.STAGE4.BLOCK = 'BASIC'
cfg.MODEL.HRNET.STAGE4.FUSE_METHOD = 'SUM'

