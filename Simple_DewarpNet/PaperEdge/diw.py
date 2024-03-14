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

class DIW(Dataset):
    def __init__(self, root_dir, is_train=True, num=0):
        super(DIW, self).__init__()
        self.is_train = is_train
        self.num = num
        # load the list of diw images
        with open('./data/diw_5k.txt', 'r') as fid:
            self.X = fid.read().splitlines()
        self.X = [root_dir + '/img/' + t + '.jpg' for t in self.X]

        # with open('./data/bgtex.txt', 'r') as fid:
        #     self.bgtex = fid.read().splitlines()

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
        bg = cv2.imread(self.bgtex[ind]).astype(np.float32) / 255.0
        bg = cv2.resize(bg, (200, 200))
        bg = np.tile(bg, (3, 3, 1))

        im = torch.from_numpy(im.transpose((2, 0, 1)).copy())
        ms = torch.from_numpy(ms.transpose((2, 0, 1)).copy())
        bg = torch.from_numpy(bg.transpose((2, 0, 1)).copy())

        return im, ms, bg

class DIWDataAug(nn.Module):
    def __init__(self):
        super(DIWDataAug, self).__init__()
        # 이미지 밝기, 대비, 채도, 색조 무작위로 적용
        self.cj = KA.ColorJitter(0.1, 0.1, 0.1, 0.1)
    
    def forward(self, img, ms, bg):
        # tight crop
        # 객체가 있는 영역만 남긴다
        # mask = ms[:, 0] > 0.5
        
        # 이미지의 갯수 뽑기, 배치사이즈
        B = img.size(0)
        # 랜덤한 정수 5개 생성
        # c = torch.randint(20, (B, 5))
        
        img_list = []
        msk_list = []
        for ii in range(B):
            # x_img = img[ii]
            # x_msk = mask[ii]
            # 객체를 포함하는 영역을 추출
            # y, x = x_msk.nonzero(as_tuple=True)
            # minx = x.min()
            # maxx = x.max()
            # miny = y.min()
            # maxy = y.max()
            # x_img = x_img[:, miny : maxy + 1, minx : maxx + 1]
            # x_msk = x_msk[None, miny : maxy + 1, minx : maxx + 1]

            # padding
            # 추출한 영역에서 랜덤한 패딩을 추가
            # 랜덤한 패딩값을 상하좌우 각각의 면에 넣으면서 찌그러짐이
            # 발생한다
            # x_img = F.pad(x_img, c[ii, : 4].tolist())
            # x_msk = F.pad(x_msk, c[ii, : 4].tolist())

            # replace bg
            # 나머지공간에 랜덤 배경을 추가한다
            # if c[ii][-1] > 2:
            #     x_bg = bg[ii][:, :x_img.size(1), :x_img.size(2)]
            # else:
            #     x_bg = torch.ones_like(x_img) * torch.rand((3, 1, 1), device=x_img.device)
            # x_msk = x_msk.float()
            # x_img = x_img * x_msk + x_bg * (1. - x_msk)

            # resize
            # 이미지와 마스크를 리사이징한다
            # 마스크는 더작게 리사이징해도 일반적으로 좋다고한다
            x_img = KG.resize(x_img[None, :], (256, 256))
            x_msk = KG.resize(x_msk[None, :], (64, 64))
            img_list.append(x_img)
            msk_list.append(x_msk)
        img = torch.cat(img_list)
        msk = torch.cat(msk_list)
        
        # jitter color
        # 위에서 정의한 대로 이미지의 색상에 대한 무작위 변환 적용
        # img = self.cj(img)
        return img, msk


