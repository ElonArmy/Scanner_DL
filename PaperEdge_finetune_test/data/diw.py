import cv2
import numpy as np
import os
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

import random

import kornia.augmentation as KA
import kornia.geometry.transform as KG

class MixDataset(torch.utils.data.Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets

    def __getitem__(self, ii):
        len_diw = len(self.datasets[1])
        jj = ii % len_diw
        # return self.datasets[1][jj]
        return self.datasets[0][ii], self.datasets[1][jj]

    def __len__(self):
        return len(self.datasets[0])

class DIW(Dataset):
    def __init__(self, root_dir, is_train=True, num=0):
        super(DIW, self).__init__()
        self.is_train = is_train
        self.num = num
        # load the list of diw images
        with open('./data/diw_test.txt', 'r') as fid:
            self.X = fid.read().splitlines()
        self.X = [root_dir + '/img/' + t for t in self.X]

        with open('./data/bgtex.txt', 'r') as fid:
            self.bgtex = fid.read().splitlines()

    def __len__(self):
        if self.num:
            return self.num
        else:
            return len(self.X)

    def __getitem__(self, index):
        t = self.X[index]
        im = cv2.imread(t).astype(np.float32) / 255.0
        im = im[..., ::-1]

        t = t.replace('img', 'seg')
        ms = cv2.imread(t).astype(np.float32) / 255.0
        ms = np.mean(ms, axis=2, keepdims=True)

        # random sample a background image
        ind = random.randint(0, len(self.bgtex) - 1)
        bg = cv2.imread('./data/bgtex/'+self.bgtex[ind]).astype(np.float32) / 255.0
        bg = cv2.resize(bg, (200, 200))
        bg = np.tile(bg, (3, 3, 1))

        im = torch.from_numpy(im.transpose((2, 0, 1)).copy())
        ms = torch.from_numpy(ms.transpose((2, 0, 1)).copy())
        bg = torch.from_numpy(bg.transpose((2, 0, 1)).copy())

        return im, ms, bg

class DIWDataAug(nn.Module):
    def __init__(self):
        super(DIWDataAug, self).__init__()
        # self.cj = KA.ColorJitter(0.1, 0.1, 0.1, 0.1)
    
    # def forward(self, img, ms, bg):
    def forward(self, img, ms):
        # tight crop
        # mask = ms[:, 0] > 0.5
        
        B = img.size(0)
        # c = torch.randint(20, (B, 5))
        
        img_list = []
        msk_list = []
        for ii in range(B):
            x_img = img[ii]
            x_msk = ms[ii]
            # y, x = x_msk.nonzero(as_tuple=True)
            # minx = x.min()
            # maxx = x.max()
            # miny = y.min()
            # maxy = y.max()
            # x_img = x_img[:, miny : maxy + 1, minx : maxx + 1]
            # x_msk = x_msk[None, miny : maxy + 1, minx : maxx + 1]

            # # padding
            # x_img = F.pad(x_img, c[ii, : 4].tolist())
            # x_msk = F.pad(x_msk, c[ii, : 4].tolist())

            # # replace bg
            # if c[ii][-1] > 2:
            #     x_bg = bg[ii][:, :x_img.size(1), :x_img.size(2)]
            # else:
            #     x_bg = torch.ones_like(x_img) * torch.rand((3, 1, 1), device=x_img.device)
            # x_msk = x_msk.float()
            # x_img = x_img * x_msk + x_bg * (1. - x_msk)
            
            x_img = x_img.unsqueeze(0)  # [1, 1, H, W] 또는 [N, C, H, W]로 변환
            x_msk = x_msk.unsqueeze(0)  # [1, 1, H, W] 또는 [N, C, H, W]로 변환

            # 각 이미지와 마스크를 원하는 크기로 리사이징
            resized_img = F.interpolate(x_img[None, :], size=(256, 256), mode='bilinear', align_corners=True)
            resized_msk = F.interpolate(x_msk[None, :], size=(64, 64), mode='bilinear', align_corners=True)
            
            img_list.append(resized_img)
            msk_list.append(resized_msk)
            
        img = torch.cat(img_list, dim=0)
        msk = torch.cat(msk_list, dim =0)
        # # jitter color
        # img = self.cj(img)
        return img, msk


