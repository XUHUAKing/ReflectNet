"""
Author: xuhua.huang
"""
from tensorpack import *
import argparse
import cv2
import numpy as np

class RealDataFlow(DataFlow):
    # currently this is a hardcode dataset
    def __init__(self, i0, i45, i90):
        self.i0 = i0
        self.i45 = i45
        self.i90 = i90
        self.lt = i0
        self.lr = i0

    def __iter__(self):
        i0 = self.i0
        i45 = self.i45
        i90 = self.i90
        lt = self.lt
        lr = self.lr
        # load data from python and return
        yield [i0, i45, i90, lt, lr]

    def __len__(self):
        return 10

    def get_data(self):
        return self.__iter__()

if __name__ == '__main__':
    pass
