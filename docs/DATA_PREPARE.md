## data prepare

It is recommended to symlink the dataset root to `$SEGMENTRON/datasets`.


```
SegmenTron
|-- configs
|-- datasets
|   |-- ade
|   |   |-- ADEChallengeData2016
|   |   |   |-- annotations
|   |   |   `-- images
|   |   |-- downloads
|   |   `-- release_test
|   |       `-- testing
|   |-- cityscapes
|   |   |-- gtFine
|   |   |   |-- test
|   |   |   |-- train
|   |   |   `-- val
|   |   `-- leftImg8bit
|   |       |-- test
|   |       |-- train
|   |       `-- val
|   |-- coco
|   |   |-- annotations
|   |   |-- train2017
|   |   `-- val2017
|   `-- voc
|       |-- VOC2007
|       |   |-- Annotations
|       |   |-- ImageSets
|       |   |-- JPEGImages
|       |   |-- SegmentationClass
|       |   `-- SegmentationObject
|       |-- VOC2012
|       |   |-- Annotations
|       |   |-- ImageSets
|       |   |-- JPEGImages
|       |   |-- SegmentationClass
|       |   `-- SegmentationObject
|       `-- VOCaug
|           |-- benchmark_code_RELEASE
|           `-- dataset
|-- docs
|-- segmentron
|-- tools

```

### cityscape
Goto [Cityscape](https://www.cityscapes-dataset.com) register a account and download datasets.

### coco

run following command, and it will automatically symlink ```your-download-dir``` to ```datasets/coco```
```
python segmentron/data/downloader/mscoco.py --download-dir your-download-dir
```

### pascal aug & voc
run following command, and it will automatically symlink ```your-download-dir``` to ```datasets/voc```
```
python segmentron/data/downloader/pascal_voc.py --download-dir your-download-dir
```

### ade20k
run following command, and it will automatically symlink ```your-download-dir``` to ```datasets/ade```
```
python segmentron/data/downloader/ade20k.py --download-dir your-download-dir
```