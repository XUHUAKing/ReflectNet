#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
Author: xuhua.huang
"""

"""
train model on own dataset
total batch size = (batch size of input source) * #GPU

python train.py --data ../../data_new/data_all_0609/20190609/Indoor/1/ --batch 32 --gpus 0

"""

import argparse
import numpy as np
import cv2
import os
from tensorpack import *
import time
import glob
from contextlib import contextmanager
from dataset import MyDataFlow
import model
import tensorflow as tf

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"  # specify which GPU(s) to be used

def main(args):
    logger.auto_set_dir()

    global HSHAPE, WSHAPE
    model.HSHAPE, model.WSHAPE = 1024, 1224

    # build dataset
    dataset_train = MyDataFlow(args.data)
    # group data into batches of size 32
    dataset_train = BatchData(dataset_train, args.batch)
    # set up gpus
    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpus  # specify which GPU(s) to be used
    gpus = [int(x) for x in args.gpus.split(',')]
    batch_size = args.batch * len(gpus)
    print('total batch size is: ', batch_size)

    steps_per_epoch = -(-len(dataset_train)//batch_size) # get ceiling of division
    config = TrainConfig(
        model=model.Model(model.HSHAPE, model.WSHAPE),
        session_init=get_model_loader('data/checkpoint'),
        # The input source for training. FeedInput is slow, this is just for demo purpose.
        # In practice it's best to use QueueInput or others. See tutorials for details.
        data=QueueInput(dataset_train),
        callbacks=[
            ModelSaver(),   # save the model after every epoch
        ],
        steps_per_epoch=steps_per_epoch,
        max_epoch=12,
    )
    trainer = SyncMultiGPUTrainerReplicated(gpus)
    launch_train_with_config(config, trainer)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', help='data path',
                        type=str, default='',required=True)
    parser.add_argument('--gpus', help='gpu ids',
                        type=str, default='0',required=True)
    parser.add_argument('--batch', help='batch size',
                        type=int, default=32)
    args = parser.parse_args()
    main(args)
