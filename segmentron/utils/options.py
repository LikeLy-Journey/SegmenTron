import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Segmentron')
    parser.add_argument('--config-file', metavar="FILE",
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
    # for evaluation
    parser.add_argument('--val-epoch', type=int, default=1,
                        help='run validation every val-epoch')
    parser.add_argument('--skip-val', action='store_true', default=False,
                        help='skip validation during training')
    # for visual
    parser.add_argument('--input-img', type=str, default='tools/demo_vis.png',
                        help='path to the input image or a directory of images')
    # config options
    parser.add_argument('opts', help='See config for all options',
                        default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()

    return args