import argparse
import copy
import os
import os.path as osp
import sys
import time

import torch
from segmentron import __version__
from segmentron.datasets import build_dataset
from segmentron.engine import launch
from segmentron.engine.trainer import IterBasedTrainer
from segmentron.models import build_segmentor
from segmentron.utils import (Config, DictAction, collect_env, comm,
                              get_git_hash, get_root_logger)
from segmentron.utils.utils import set_random_seed


def parse_args():
    parser = argparse.ArgumentParser(description='Train a segmentor')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--load-from', help='the checkpoint file to load weights from')
    parser.add_argument(
        '--resume-from', help='the checkpoint file to resume from')
    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='whether not to evaluate the checkpoint during training')
    parser.add_argument(
        '--eval-only', action='store_true', help='perform evaluation only')
    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument(
        '--gpus',
        type=int,
        default=1,
        help='number of gpus to use '
        '(only applicable to non-distributed training)')
    parser.add_argument(
        '--num-machines', type=int, default=1, help='total number of machines')
    parser.add_argument(
        '--machine-rank',
        type=int,
        default=0,
        help='the rank of this machine (unique per machine)')

    # PyTorch still may leave orphan processes in multi-gpu training.
    # Therefore we use a deterministic way to obtain port,
    # so that users are aware of orphan processes by seeing the port occupied.
    port = 2**15 + 2**14 + hash(
        os.getuid() if sys.platform != 'win32' else 1) % 2**14
    parser.add_argument(
        '--dist-url',
        default='auto',
        # default="tcp://127.0.0.1:{}".format(port),
        help='initialization URL for pytorch distributed backend. See '
        'https://pytorch.org/docs/stable/distributed.html for details.',
    )
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--options', nargs='+', action=DictAction, help='custom options')

    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def main(args):
    cfg = Config.fromfile(args.config)
    if args.options is not None:
        cfg.merge_from_dict(args.options)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])
    if args.load_from is not None:
        cfg.load_from = args.load_from
    if args.resume_from is not None:
        cfg.resume_from = args.resume_from

    cfg.gpu_ids = range(args.gpus)

    distributed = comm.get_world_size() > 1
    # create work_dir
    # mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    os.makedirs(osp.abspath(cfg.work_dir), exist_ok=True)
    # dump config
    cfg.dump(osp.join(cfg.work_dir, osp.basename(args.config)))
    # init the logger before other steps
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)

    # init the meta dict to record some important information such as
    # environment info and seed, which will be logged
    meta = dict()
    # log env info
    env_info_dict = collect_env()
    env_info = '\n'.join([f'{k}: {v}' for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'
    logger.info('Environment info:\n' + dash_line + env_info + '\n' +
                dash_line)
    meta['env_info'] = env_info

    # log some basic info
    logger.info(f'Distributed training: {distributed}')
    logger.info(f'Config:\n{cfg.pretty_text}')

    # set random seeds
    if args.seed is not None:
        logger.info(f'Set random seed to {args.seed}, deterministic: '
                    f'{args.deterministic}')
        set_random_seed(args.seed, deterministic=args.deterministic)
    cfg.seed = args.seed
    meta['seed'] = args.seed
    meta['exp_name'] = osp.basename(args.config)

    if cfg.checkpoint_config is not None:
        # save mmseg version, config file content and class names in
        # checkpoints as meta data
        cfg.checkpoint_config.meta = dict(
            mmseg_version=f'{__version__}+{get_git_hash()[:7]}',
            config=cfg.pretty_text)

    trainer = IterBasedTrainer(
        cfg, logger=logger, meta=meta, eval_only=args.eval_only)
    trainer.timestamp = timestamp

    if cfg.resume_from:
        trainer.resume(cfg.resume_from)
    elif cfg.load_from:
        trainer.load_checkpoint(cfg.load_from)
    trainer.run()


if __name__ == '__main__':
    args = parse_args()
    print('Command Line Args:', args)
    launch(
        main,
        args.gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args, ),
    )
