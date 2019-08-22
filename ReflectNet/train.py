#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
Authors: Patrick Wieschollek, Orazio Gallo, Jinwei Gu, and Jan Kautz
"""

"""
evaluate model on given image

prepare data:
---------------------
cd example
dcraw -v -W -g 1 1 -6 *.ARW
cd ..
i0=example/DSC01908.ppm
i45=example/DSC01909.ppm
i90=example/DSC01910.ppm
prefix=bar
scale=0.25
python eval.py --scale ${scale} --i0 ${i0} --i45 ${i45} --i90 ${i90} --out example/ --prefix ${prefix}

"""

import argparse
import numpy as np
import cv2
import os
from tensorpack import *
import time
import glob
from contextlib import contextmanager
from dataset import RealDataFlow
import model
import tensorflow as tf

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"  # specify which GPU(s) to be used

@contextmanager
def benchmark(name="unnamed context"):
    """timing
    Args:
        name (str): output-name for timing
    Example:
        with meta.benchmark('doing heavy stuff right now tooks'):
            sleep(1)
    """
    elapsed = time.time()
    yield
    elapsed = time.time() - elapsed
    print('[{}] finished in {} ms'.format(name, int(elapsed * 1000)))


def main(args):
    logger.auto_set_dir()
    assert os.path.isdir(args.out)

    if not os.path.isfile(args.i0):
        ppms = glob.glob(args.out + "/*.ppm")
        ppms = sorted(ppms)
        I0 = cv2.imread(ppms[0]).astype(np.float32) / 255.
        I45 = cv2.imread(ppms[1]).astype(np.float32) / 255.
        I90 = cv2.imread(ppms[2]).astype(np.float32) / 255.
    else:
        print("****path file exist, the path is correct")
        assert os.path.isfile(args.i0)
        assert os.path.isfile(args.i45)
        assert os.path.isfile(args.i90)
        I0 = cv2.imread(args.i0).astype(np.float32) / 255.
        I45 = cv2.imread(args.i45).astype(np.float32) / 255.
        I90 = cv2.imread(args.i90).astype(np.float32) / 255.

    if args.scale is not 1.0:
        I0 = cv2.resize(I0, (0, 0), fx=args.scale, fy=args.scale)
        I45 = cv2.resize(I45, (0, 0), fx=args.scale, fy=args.scale)
        I90 = cv2.resize(I90, (0, 0), fx=args.scale, fy=args.scale)

    h, w, _ = I0.shape
    h = (h // 8) * 8
    w = (w // 8) * 8

    # I0 = tf.tile(I0,[1,1,3])
    # I45 = tf.tile(I45,[1,1,3])
    # I90 = tf.tile(I90,[1,1,3])

    I0 = I0[:h, :w, :]
    I45 = I45[:h, :w, :]
    I90 = I90[:h, :w, :]

    global HSHAPE, WSHAPE
    model.HSHAPE, model.WSHAPE = I0.shape[:2]

    I0 = I0[None, :, :, :]
    I45 = I45[None, :, :, :]
    I90 = I90[None, :, :, :]
    # (1, 1000, 1504, 9)

    # build dataset
    dataset_train = RealDataFlow(I0, I45, I90)

    steps_per_epoch = len(dataset_train)
    config = TrainConfig(
        model=model.Model(model.HSHAPE, model.WSHAPE),
        session_init=get_model_loader('data/checkpoint'),
        # The input source for training. FeedInput is slow, this is just for demo purpose.
        # In practice it's best to use QueueInput or others. See tutorials for details.
        data=FeedInput(dataset_train),
        callbacks=[
            ModelSaver(),   # save the model after every epoch
        ],
        steps_per_epoch=steps_per_epoch,
        max_epoch=12,
    )
    launch_train_with_config(config, SimpleTrainer())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--scale', help='scaling of inputs',
                        type=float, default=1.0)
    parser.add_argument('--out', help='output dir', type=str, default='/tmp')
    parser.add_argument('--prefix', help='prefix', type=str, required=True)
    parser.add_argument(
        '--i0', help='image at 0 polarization', type=str, default='')
    parser.add_argument(
        '--i45', help='image at 45 polarization', type=str, default='')
    parser.add_argument(
        '--i90', help='image at 90 polarization', type=str, default='')
    args = parser.parse_args()
    main(args)
