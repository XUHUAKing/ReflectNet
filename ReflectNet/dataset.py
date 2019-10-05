"""
Author: xuhua.huang
"""
from tensorpack import *
from glob import glob
import argparse
import cv2
import numpy as np
import tensorflow as tf
import scipy
import os

class MyDataFlow(DataFlow):
    def __init__(self, data_path=None):
        if not os.path.isdir(data_path):
            raise Exception('data path not exist')
        self.items = []
        img_names = glob(data_path+'/*.npy')
        img_names.sort()
        mask_names = glob(data_path+'/mask' + '/*mask.png')
        mask_names.sort()
        for tmp_mask in mask_names:
            tmp_M = data_path + '/IMG_' + tmp_mask[-18:-14] + '.npy'
            tmp_R = data_path + '/IMG_' + tmp_mask[-13:-9] + '.npy'
            if os.path.isfile(tmp_M) and os.path.isfile(tmp_R):
                self.items.append([tmp_M,tmp_R,tmp_mask])
            else:
                print(tmp_M, tmp_R, tmp_mask, 'not exist...')
                raise Exception('M/R/Mask not exist')
        print("Data load succeed!")

    def __iter__(self):
        for item in self.items:
            M_name, R_name, mask_name = item
            tmp_M = np.load(M_name) # (1, 1024, 1224, 5) = 1*H*W*5
            tmp_R = np.load(R_name) # (1, 1024, 1224, 5)
            tmp_mask=scipy.misc.imread(mask_name,'L')[::2,::2,np.newaxis]/255.
            # get polarized image
            i0 = tmp_M[0,:,:,0]
            i45 = tmp_M[0,:,:,1]
            i90 = tmp_M[0,:,:,2]
            # crop glass area
            i0 = i0[:,:,np.newaxis]*tmp_mask
            i45 = i45[:,:,np.newaxis]*tmp_mask
            i90 = i90[:,:,np.newaxis]*tmp_mask
            # duplicate along RGB channel
            I0 = np.tile(i0,[1,1,3])
            I45 = np.tile(i45,[1,1,3])
            I90 = np.tile(i90,[1,1,3])
            # get lt and lr
            M = 0.5*tmp_M[0,:,:,-1]
            R = 0.5*tmp_R[0,:,:,-1]
            T = M - R
            # crop glass area
            T = T[:,:,np.newaxis]*tmp_mask
            T[T<0] = 0
            R = R[:,:,np.newaxis]*tmp_mask
            # duplicate along RGB channel
            lt = np.tile(T,[1,1,3])
            lr = np.tile(R,[1,1,3])
            yield[I0, I45, I90, lt, lr]

    def __len__(self):
        print(len(self.items))
        return len(self.items)

    def get_data(self):
        return self.__iter__()

if __name__ == '__main__':
    pass
