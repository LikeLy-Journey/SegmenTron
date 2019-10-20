import argparse
import time
import datetime
import os
import shutil
import sys

cur_path = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.split(cur_path)[0]
sys.path.append(root_path)

import logging
import pprint
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

from torchvision import transforms
from segmentron.data.dataloader import get_segmentation_dataset
from segmentron.models.model_zoo import get_segmentation_model
from segmentron.utils.loss import get_segmentation_loss
from segmentron.utils.distributed import *
from segmentron.utils.logger import setup_logger
from segmentron.utils.optimizer import get_optimizer
from segmentron.utils.lr_scheduler import get_scheduler
from segmentron.utils.score import SegmentationMetric
from segmentron.utils.config import cfg

def parse_args():
    parser = argparse.ArgumentParser(description='Semantic Segmentation Training With Pytorch')
    parser.add_argument('config_file',
                        help='config file path')
    # cuda setting
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--local_rank', type=int, default=0)
    # checkpoint and log
    parser.add_argument('--resume', type=str, default=None,
                        help='put the path to resuming file if needed')
    parser.add_argument('--log-iter', type=int, default=10,
                        help='print log every log-iter')
    # evaluation only
    parser.add_argument('--val-epoch', type=int, default=1,
                        help='run validation every val-epoch')
    parser.add_argument('--skip-val', action='store_true', default=False,
                        help='skip validation during training')
    args = parser.parse_args()

    return args


class Trainer(object):
    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device)

        # image transform
        input_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(cfg.MEAN, cfg.STD),
        ])
        # dataset and dataloader
        data_kwargs = {'transform': input_transform, 'base_size': cfg.TRAIN_BASE_SIZE,
                       'crop_size': cfg.TRAIN_CROP_SIZE}
        train_dataset = get_segmentation_dataset(cfg.DATASET, split='train', mode='train', **data_kwargs)
        val_dataset = get_segmentation_dataset(cfg.DATASET, split='val', mode='testval', **data_kwargs)
        self.iters_per_epoch = len(train_dataset) // (args.num_gpus * cfg.BATCH_SIZE)
        self.max_iters = cfg.EPOCHS * self.iters_per_epoch

        train_sampler = make_data_sampler(train_dataset, shuffle=True, distributed=args.distributed)
        train_batch_sampler = make_batch_data_sampler(train_sampler, cfg.BATCH_SIZE, self.max_iters, drop_last=True)
        val_sampler = make_data_sampler(val_dataset, False, args.distributed)
        val_batch_sampler = make_batch_data_sampler(val_sampler, cfg.BATCH_SIZE, drop_last=False)

        self.train_loader = data.DataLoader(dataset=train_dataset,
                                            batch_sampler=train_batch_sampler,
                                            num_workers=cfg.WORKERS,
                                            pin_memory=True)
        self.val_loader = data.DataLoader(dataset=val_dataset,
                                          batch_sampler=val_batch_sampler,
                                          num_workers=cfg.WORKERS,
                                          pin_memory=True)

        # create network
        self.model = get_segmentation_model().to(self.device)
        if args.distributed and cfg.TRAIN.SYNC_BATCH_NORM:
            self.model = nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
            logging.info('SyncBatchNorm is effective!')
        else:
            logging.info('Do not use SyncBatchNorm!')

        # resume checkpoint if needed
        if args.resume:
            if os.path.isfile(args.resume):
                name, ext = os.path.splitext(args.resume)
                assert ext == '.pkl' or '.pth', 'Sorry only .pth and .pkl files supported.'
                logging.info('Resuming training, loading {}...'.format(args.resume))
                self.model.load_state_dict(torch.load(args.resume, map_location=lambda storage, loc: storage))

        # create criterion
        self.criterion = get_segmentation_loss(cfg.MODEL.MODEL_NAME, use_ohem=cfg.SOLVER.OHEM, aux=cfg.SOLVER.AUX,
                                               aux_weight=cfg.SOLVER.AUX_WEIGHT, ignore_index=cfg.IGNORE_INDEX).to(self.device)

        # optimizer, for model just includes encoder, decoder(head and auxlayer).
        self.optimizer = get_optimizer(self.model)

        # lr scheduling
        self.lr_scheduler = get_scheduler(self.optimizer, max_iters=self.max_iters)

        if args.distributed:
            self.model = nn.parallel.DistributedDataParallel(self.model, device_ids=[args.local_rank],
                                                             output_device=args.local_rank,
                                                             find_unused_parameters=True)

        # evaluation metrics
        self.metric = SegmentationMetric(train_dataset.num_class, args.distributed)
        self.best_pred = 0.0


    def train(self):
        save_to_disk = get_rank() == 0
        epochs, max_iters, iters_per_epoch = cfg.EPOCHS, self.max_iters, self.iters_per_epoch
        log_per_iters, val_per_iters = self.args.log_iter, self.args.val_epoch * self.iters_per_epoch
        save_per_iters = cfg.TRAIN.SNAPSHOT_EPOCH * self.iters_per_epoch
        start_time = time.time()
        logging.info('Start training, Total Epochs: {:d} = Total Iterations {:d}'.format(epochs, max_iters))

        self.model.train()
        for iteration, (images, targets, _) in enumerate(self.train_loader):
            epoch = iteration // iters_per_epoch + 1
            iteration = iteration + 1

            images = images.to(self.device)
            targets = targets.to(self.device)

            outputs = self.model(images)
            loss_dict = self.criterion(outputs, targets)

            losses = sum(loss for loss in loss_dict.values())

            # reduce losses over all GPUs for logging purposes
            loss_dict_reduced = reduce_loss_dict(loss_dict)
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())

            self.optimizer.zero_grad()
            losses.backward()
            self.optimizer.step()
            self.lr_scheduler.step()

            eta_seconds = ((time.time() - start_time) / iteration) * (max_iters - iteration)
            eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

            if iteration % log_per_iters == 0 and save_to_disk:
                logging.info(
                    "Epoch: {:d}/{:d} || Iters: {:d}/{:d} || Lr: {:.6f} || "
                    "Loss: {:.4f} || Cost Time: {} || Estimated Time: {}".format(
                        epoch, epochs, iteration % iters_per_epoch, iters_per_epoch,
                        self.optimizer.param_groups[0]['lr'], losses_reduced.item(),
                        str(datetime.timedelta(seconds=int(time.time() - start_time))),
                        eta_string))

            if iteration % save_per_iters == 0 and save_to_disk:
                save_checkpoint(self.model, epoch, is_best=False)

            if not self.args.skip_val and iteration % val_per_iters == 0:
                self.validation(epoch)
                self.model.train()

        save_checkpoint(self.model, "final", is_best=False)
        total_training_time = time.time() - start_time
        total_training_str = str(datetime.timedelta(seconds=total_training_time))
        logging.info(
            "Total training time: {} ({:.4f}s / it)".format(
                total_training_str, total_training_time / max_iters))

    def validation(self, epoch):
        self.metric.reset()
        if self.args.distributed:
            model = self.model.module
        else:
            model = self.model
        torch.cuda.empty_cache()
        model.eval()
        for i, (image, target, filename) in enumerate(self.val_loader):
            image = image.to(self.device)
            target = target.to(self.device)

            with torch.no_grad():
                size = image.size()[2:]
                if size[0] < cfg.EVAL_CROP_SIZE[0] and size[1] < cfg.EVAL_CROP_SIZE[1]:
                    pad_height = cfg.EVAL_CROP_SIZE[0] - size[0]
                    pad_width = cfg.EVAL_CROP_SIZE[1] - size[1]
                    image = F.pad(image, (0, pad_height, 0, pad_width))
                    output = model(image)[0]
                    output = output[..., :size[0], :size[1]]
                else:
                    logging.info('no eval pad!')
                    output = model(image)[0]
            self.metric.update(output, target)
            pixAcc, mIoU = self.metric.get()
            logging.info("[EVAL] Sample: {:d}, pixAcc: {:.3f}, mIoU: {:.3f}".format(i + 1, pixAcc, mIoU))
        pixAcc, mIoU = self.metric.get()
        logging.info("[EVAL END] Epoch: {:d}, pixAcc: {:.3f}, mIoU: {:.3f}".format(epoch, pixAcc, mIoU))
        synchronize()


def save_checkpoint(model, epoch, is_best=False):
    """Save Checkpoint"""
    directory = os.path.expanduser(cfg.TRAIN.MODEL_SAVE_DIR)
    directory = os.path.join(directory, '{}_{}_{}_{}'.format(cfg.MODEL.MODEL_NAME, cfg.MODEL.BACKBONE,
                                                             cfg.DATASET, cfg.TIME_STAMP))
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = '{}.pth'.format(str(epoch))
    filename = os.path.join(directory, filename)

    # if args.distributed:
    #     model = model.module
    torch.save(model.module.state_dict() if hasattr(model, 'module') else model.state_dict(), filename)
    if is_best:
        best_filename = '{}_best_model.pth'.format(str(epoch))
        best_filename = os.path.join(directory, best_filename)
        shutil.copyfile(filename, best_filename)


if __name__ == '__main__':
    args = parse_args()
    cfg.update_from_file(args.config_file)
    cfg.check_and_infer()
    # if args.opts is not None:
    #     cfg.update_from_list(args.opts)

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.num_gpus = num_gpus
    args.distributed = num_gpus > 1

    if not args.no_cuda and torch.cuda.is_available():
        cudnn.benchmark = True
        args.device = "cuda"
    else:
        args.distributed = False
        args.device = "cpu"
    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        synchronize()

    setup_logger("Segmentron", cfg.TRAIN.LOG_SAVE_DIR, get_rank(), filename='{}_{}_{}_{}_log.txt'.format(
        cfg.MODEL.MODEL_NAME, cfg.MODEL.BACKBONE, cfg.DATASET, cfg.TIME_STAMP))
    logging.info("Using {} GPUs".format(num_gpus))
    logging.info(args)
    logging.info(pprint.pformat(cfg))
    trainer = Trainer(args)
    trainer.train()
    torch.cuda.empty_cache()
