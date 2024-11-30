# PyTorch for Semantic Segmentation

## What's new
- Implemented backbone code of swin transformer from detectron2.
- Implemented model UPerNet from mmsegmentation along with pretrained checkpoints.
- Implemented model MaskFormer from detectron2 along with pretrained checkpoints.

## Introduce
This repository contains some models for semantic segmentation and the pipeline of training and testing models, 
implemented in PyTorch.


![](docs/images/demo.png)

## Model zoo

|          Model          |  Backbone  |    Datasets     |  eval size  | Mean IoU(paper) |                                                       Mean IoU(this repo)                                                        |
|:-----------------------:|:----------:|:---------------:|:-----------:|:---------------:|:--------------------------------------------------------------------------------------------------------------------------------:|
|     DeepLabv3_plus      | xception65 | cityscape(val)  | (1025,2049) |      78.8       |      [78.93](https://github.com/LikeLy-Journey/SegmenTron/releases/download/v0.1.0/deeplabv3_plus_xception_segmentron.pth)       |
|     DeepLabv3_plus      | xception65 |    coco(val)    |   480/520   |        -        |    [70.50](https://github.com/LikeLy-Journey/SegmenTron/releases/download/v0.1.0/deeplabv3_plus_xception_coco_segmentron.pth)    |
|     DeepLabv3_plus      | xception65 | pascal_aug(val) |   480/520   |        -        | [89.56](https://github.com/LikeLy-Journey/SegmenTron/releases/download/v0.1.0/deeplabv3_plus_xception_pascal_aug_segmentron.pth) |
|     DeepLabv3_plus      | xception65 | pascal_voc(val) |   480/520   |        -        | [88.39](https://github.com/LikeLy-Journey/SegmenTron/releases/download/v0.1.0/deeplabv3_plus_xception_pascal_voc_segmentron.pth) |
|     DeepLabv3_plus      | resnet101  | cityscape(val)  | (1025,2049) |        -        |      [78.27](https://github.com/LikeLy-Journey/SegmenTron/releases/download/v0.1.0/deeplabv3_plus_resnet101_segmentron.pth)      |
|          Danet          | resnet101  | cityscape(val)  | (1024,2048) |      79.9       |              [79.34](https://github.com/LikeLy-Journey/SegmenTron/releases/download/v0.1.0/danet101_segmentron.pth)              |
|         Pspnet          | resnet101  | cityscape(val)  | (1025,2049) |      78.63      |          [77.00](https://github.com/LikeLy-Journey/SegmenTron/releases/download/v0.1.0/pspnet_resnet101_segmentron.pth)          |
|         UPerNet         |   Swin-b   |   ade20k(val)   |  (480,480)  |      51.6       |            [49.12](https://github.com/juliantang324/SegmenTron/releases/download/v0.1.0/upernet_swinb_segmentron.pth)            |
|       MaskFormer        |   Swin-t   |   ade20k(val)   |  (480,480)  |      48.8       |          [47.11](https://github.com/juliantang324/SegmenTron/releases/download/v0.1.0/maskformer_swint_segmentron.pth)           |
|       MaskFormer        |   Swin-b   |   ade20k(val)   |  (480,480)  |      52.3       |          [51.87](https://github.com/juliantang324/SegmenTron/releases/download/v0.1.0/maskformer_swinb_segmentron.pth)           |

### real-time models
|     Model      |   Backbone    |    Datasets    |  eval size  | Mean IoU(paper) |                                                   Mean IoU(this repo)                                                   |  FPS   |
|:--------------:|:-------------:|:--------------:|:-----------:|:---------------:|:-----------------------------------------------------------------------------------------------------------------------:|:------:|
|     ICnet      | resnet50(0.5) | cityscape(val) | (1024,2048) |      67.8       |                                                            -                                                            | 41.39  |
| DeepLabv3_plus |  mobilenetV2  | cityscape(val) | (1024,2048) |      70.7       | [70.3](https://github.com/LikeLy-Journey/SegmenTron/releases/download/v0.1.0/deeplabv3_plus_mobilenetv2_segmentron.pth) | 46.64  |
|    BiSeNet     |   resnet18    | cityscape(val) | (1024,2048) |        -        |                                                            -                                                            | 39.90  |
|     LEDNet     |       -       | cityscape(val) | (1024,2048) |        -        |                                                            -                                                            | 31.78  |
|     CGNet      |       -       | cityscape(val) | (1024,2048) |        -        |                                                            -                                                            | 46.11  |
|    HardNet     |       -       | cityscape(val) | (1024,2048) |      75.9       |                                                            -                                                            | 69.06  |
|     DFANet     |   xceptionA   | cityscape(val) | (1024,2048) |      70.3       |                                                            -                                                            | 21.46  |
|     HRNet      | w18_small_v1  | cityscape(val) | (1024,2048) |      70.3       |     [70.5](https://github.com/LikeLy-Journey/SegmenTron/releases/download/v0.1.0/hrnet_w18_small_v1_segmentron.pth)     | 66.01  |
|   Fast_SCNN    |       -       | cityscape(val) | (1024,2048) |      68.3       |         [68.9](https://github.com/LikeLy-Journey/SegmenTron/releases/download/v0.1.0/fast_scnn_segmentron.pth)          | 145.77 |

FPS was tested on V100.

## Environments

- python 3
- torch >= 1.1.0
- torchvision
- pyyaml
- Pillow
- numpy

## INSTALL

```
python setup.py develop
```

if you do not want to run CCNet, you do not need to install, just comment following line in ```segmentron/models/__init__.py```
```
from .ccnet import CCNet
```
## Dataset prepare
Support cityscape, coco, voc, ade20k now.

Please refer to [DATA_PREPARE.md](docs/DATA_PREPARE.md) for dataset preparation.

## Pretrained backbone models 

pretrained backbone models will be download automatically in pytorch default directory(```~/.cache/torch/checkpoints/```).

## Code structure
```
├── configs    # yaml config file
├── segmentron # core code
├── tools      # train eval code
└── datasets   # put datasets here 
```

## Train
### Train with a single GPU
```
CUDA_VISIBLE_DEVICES=0 python -u tools/train.py --config-file configs/cityscapes_deeplabv3_plus.yaml
```
### Train with multiple GPUs

Auto-mixed precision is turned on by default. To turn it off, add argument "--use-amp".

```
CUDA_VISIBLE_DEVICES=0,1,2,3 ./tools/dist_train.sh ${CONFIG_FILE} ${GPU_NUM} [optional arguments]
```

## Eval
### Eval with a single GPU
You can download trained model from model zoo table above, or train by yourself.
```
CUDA_VISIBLE_DEVICES=0 python -u ./tools/eval.py --config-file configs/cityscapes_deeplabv3_plus.yaml \
TEST.TEST_MODEL_PATH your_test_model_path

```
### Eval with a multiple GPUs
```
CUDA_VISIBLE_DEVICES=0,1,2,3 ./tools/dist_test.sh ${CONFIG_FILE} ${GPU_NUM} \
TEST.TEST_MODEL_PATH your_test_model_path
```

## References
- [PyTorch-Encoding](https://github.com/zhanghang1989/PyTorch-Encoding)
- [detectron2](https://github.com/facebookresearch/detectron2)
- [gloun-cv](https://github.com/dmlc/gluon-cv)
- [mmsegmentation](https://github.com/open-mmlab/mmsegmentation)
- [MaskFormer](https://github.com/facebookresearch/MaskFormer)
