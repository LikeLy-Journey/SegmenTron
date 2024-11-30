"""Prepare ADE20K dataset"""
import os
import sys

cur_path = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.split(os.path.split(os.path.split(cur_path)[0])[0])[0]
sys.path.append(root_path)

import argparse
import zipfile
from segmentron.utils import download, makedirs


def parse_args():
    parser = argparse.ArgumentParser(
        description='Initialize ADE20K dataset.',
        epilog='Example: python setup_ade20k.py',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--download-dir', default=None, help='dataset directory on disk')
    args = parser.parse_args()
    return args

def download_ade(path, overwrite=False):
    _AUG_DOWNLOAD_URLS = [
        ('http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip',
         '219e1696abb36c8ba3a3afe7fb2f4b4606a897c7'),
        ('http://data.csail.mit.edu/places/ADEchallenge/release_test.zip',
         'e05747892219d10e9243933371a497e905a4860c'),
    ]
    download_dir = os.path.join(path, 'downloads')
    makedirs(download_dir)
    for url, checksum in _AUG_DOWNLOAD_URLS:
        filename = download(url, path=download_dir, overwrite=overwrite, sha1_hash=checksum)
        # extract
        with zipfile.ZipFile(filename,"r") as zip_ref:
            zip_ref.extractall(path=path)


if __name__ == '__main__':
    args = parse_args()
    default_dir = os.path.join(root_path, 'datasets/ade')
    if args.download_dir is not None:
        _TARGET_DIR = args.download_dir
    else:
        _TARGET_DIR = default_dir
    makedirs(_TARGET_DIR)

    try:
        os.symlink(_TARGET_DIR, default_dir)
    except Exception as e:
        print(e)
    download_ade(_TARGET_DIR, overwrite=False)
