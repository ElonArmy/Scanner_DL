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

import matplotlib.pyplot as plt

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

# test용
class DIW_test(Dataset):
    def __init__(self, root_dir, is_train=True, num=0):
        super(DIW_test, self).__init__()
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
        print(t)
        im = cv2.imread(t).astype(np.float32) / 255.0
        im = im[..., ::-1]

        t = t.replace('img', 'seg')
        ms = cv2.imread(t).astype(np.float32) / 255.0
        ms = np.mean(ms, axis=2, keepdims=True)

        # random sample a background image
        ind = random.randint(0, len(self.bgtex) - 1)
        print('./data/bgtex/'+self.bgtex[ind])
        bg = cv2.imread('./data/bgtex/'+self.bgtex[ind]).astype(np.float32) / 255.0
        bg = cv2.resize(bg, (200, 200))
        bg = np.tile(bg, (3, 3, 1))

        im = torch.from_numpy(im.transpose((2, 0, 1)).copy())
        ms = torch.from_numpy(ms.transpose((2, 0, 1)).copy())
        bg = torch.from_numpy(bg.transpose((2, 0, 1)).copy())

        return im, ms, bg

# test용
class DIWDataAug_test(nn.Module):
    def __init__(self):
        super(DIWDataAug_test, self).__init__()
        self.cj = KA.ColorJitter(0.1, 0.1, 0.1, 0.1)
    
    def forward(self, img, ms):
    # def forward(self, img, ms, bg):
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

# origin
class DIW(Dataset):
    def __init__(self, root_dir, is_train=True, num=0):
        super(DIW, self).__init__()
        self.is_train = is_train
        self.num = num
        # load the list of diw images
        with open('./data/diw_5k.txt', 'r') as fid:
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
        print(t)
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
    
#origin
class DIWDataAug(nn.Module):
    def __init__(self):
        super(DIWDataAug, self).__init__()
        self.cj = KA.ColorJitter(0.1, 0.1, 0.1, 0.1)
    
    def forward(self, img, ms, bg):
        B = img.size(0)
        # 패딩값 지정해주기randint(최소, 최대, (B, 갯수))
        c = torch.randint(50, 200, (B, 6))
        img_list = []
        msk_list = []
        
        # torch.Size([3, 512, 512]) torch.Size([512, 512])
        # torch.Size([1, 3, 256, 256]) torch.Size([1, 1, 64, 64])
        # 원본 이미지도 학습
        for ii in range(B):
            x_img = img[ii].unsqueeze(0)
            x_msk = ms[ii].unsqueeze(1)
            # print(x_img.shape, x_msk.shape)
            
            # resize
            x_img = F.interpolate(x_img, size=(256, 256), mode='bilinear', align_corners=False)
            x_msk = F.interpolate(x_msk, size=(64, 64), mode='nearest')

            img_list.append(x_img)
            msk_list.append(x_msk)
            
        # tight crop : boolean으로 바꿈
        mask = ms[:, 0] > 0.5
            
        # 원본이미지를 증강한것을 학습
        for ii in range(B):
            x_img = img[ii]
            x_msk = mask[ii]
            
            y, x = x_msk.nonzero(as_tuple=True)
            minx = x.min()
            maxx = x.max()
            miny = y.min()
            maxy = y.max()
            x_img = x_img[:, miny : maxy + 1, minx : maxx + 1]
            x_msk = x_msk[None, miny : maxy + 1, minx : maxx + 1]

            # padding
            x_img = F.pad(x_img, c[ii, : 4].tolist())
            x_msk = F.pad(x_msk, c[ii, : 4].tolist())

            # replace bg
            if c[ii][-1] > 2:
                x_bg = bg[ii][:, :x_img.size(1), :x_img.size(2)]
            else:
                x_bg = torch.ones_like(x_img) * torch.rand((3, 1, 1), device=x_img.device)
            x_msk = x_msk.float()
            
            # x_bg를 x_img와 같은 크기로 리사이징
            if x_bg.size(1) != x_img.size(1) or x_bg.size(2) != x_img.size(2):
                x_bg = F.interpolate(x_bg.unsqueeze(0), size=x_img.size()[1:], mode='bilinear', align_corners=False).squeeze(0)

            x_img = x_img * x_msk + x_bg * (1. - x_msk)

            # resize
            x_img = KG.resize(x_img[None, :], (256, 256))
            x_msk = KG.resize(x_msk[None, :], (64, 64))
            
            img_list.append(x_img)
            msk_list.append(x_msk)
        img = torch.cat(img_list)
        msk = torch.cat(msk_list)
        
        # 이미지를 시각화: 학습할 원본+증강이미지 확인해보려면
        # for ix in range(len(img)):
        #     plt.figure(figsize=(8,4))
        #     imgs = img[ix]
        #     imgs = imgs.permute(1, 2, 0).numpy()
        #     plt.subplot(1,2,1)
        #     plt.imshow(imgs)
        #     msks = msk[ix]
        #     msks = msks.permute(1, 2, 0).numpy()
        #     plt.subplot(1,2,2)
        #     plt.imshow(msks, cmap='gray')
        #     plt.show()
        # plt.close()
        # return
        
        # jitter color
        img = self.cj(img)
        return img, msk

