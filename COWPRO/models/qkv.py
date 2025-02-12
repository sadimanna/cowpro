#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import nrrd
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np
import nibabel as nib
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
from torch.utils.data import Dataset, DataLoader
import random
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
import cv2
import os
from skimage import segmentation,color
import skimage
import SimpleITK as sitk
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import torch.nn.functional as F
from torch.autograd import Variable
from skimage.future import graph
import numpy as np
from math import exp
from numpy.random import RandomState
import kornia as K
#np.random = RandomState()




class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1): 
        # inputs[inputs>0.05] = 1
        # targets[targets>0.05] = 1   
        inputs = torch.reshape(inputs,(-1,))
        targets = torch.reshape(targets,(-1,))

        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        return 1 - dice


class SegNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.theta_1 = encoder(in_channels = 1, base_c= 64)
        self.theta_2 = encoder(in_channels = 1,base_c = 64)
        self.xi = encoder(in_channels = 1,base_c = 64)
        self.decoder = decoder(256,self.theta_1,self.theta_2,self.xi,sigmoid=True)
        self.loss = DiceLoss()
        self.bceloss = nn.BCELoss()
        # self.transform = transforms.ColorJitter(brightness=0.4,contrast=0.4,saturation=0.4,hue=0.2)
        #self.prewitt = Prewitt()

    def return_mask_loss(self,feat_x,feat_y,mask_x,mask_y=None,return_loss=True):
        # x - support
        # y - query
        pred_mask_y = self.decoder(feat_x,mask_x,feat_y)

        if return_loss:
            loss = torch.log(torch.cosh(self.loss(pred_mask_y,mask_y))) + self.bceloss(pred_mask_y,mask_y)
            return pred_mask_y,loss

        else:
            return pred_mask_y


    def forward(self,feat_x,mask_x,feat_y=None,mask_y=None,epoch=None):
        '''
        feat_x --> [B, C, H, W]
        mask_x --> [B, 1, H, W]
        '''
        B,C,H,W = feat_x.shape
        if self.training:
            pred_mask_y,normal_loss = self.return_mask_loss(feat_x,feat_y,mask_x,mask_y,True)
            # _,normal_loss_inv = self.return_mask_loss(feat_y,feat_x,mask_y,mask_x)
            return pred_mask_y,mask_y,normal_loss #,normal_loss_inv
    
        else:
            with torch.no_grad():
                pred_mask_y,loss = self.return_mask_loss(feat_x,feat_y,mask_x,mask_y,True)
            return pred_mask_y,loss



    

