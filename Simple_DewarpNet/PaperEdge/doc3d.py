import cv2
import numpy as np
import scipy.interpolate
import os
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import random
import time

# HDF5 파일
from hdf5storage import loadmat

# 데이터증강관련
import kornia.augmentation as KA
import kornia.geometry.transform as KG

# 데이터셋 읽어오기
class Doc3D(Dataset):
    def __init__(self, root_dir, is_train=True, num=0):
        super(Doc3D, self).__init__()
        # self.is_train = is_train
        self.num = num
        
        # 훈련모드일 경우 doc3d_trn 리스트를 불러온다
        if is_train:
            with open('./doc3d_trn.txt', 'r') as fid:
                self.X = fid.read().splitlines()
        # 검증모드이면 doc3d_val
        else:  
            with open('./doc3d_val.txt', 'r') as fid:
                self.X = fid.read().splitlines()
        # 파일경로들을 X에 저장
        self.X = [root_dir + '/img/' + t + '.png' for t in self.X]
        
        # 배경이미지 경로리스트 에서 불러오는 변수 초기화
        with open('./bgtex.txt', 'r') as fid:
            self.bgtex = fid.read().splitlines()   # 배경이미지경로리스트   

    #데이터 길이
    def __len__(self):
        if self.num:
            return self.num
        else:
            return len(self.X)

    def __getitem__(self, index):
        # index의 이미지경로를 가져온다
        t = self.X[index]
        #이미지 파일을 불러오면서 픽셀값을 0~1 범위로 정규화한다
        im = cv2.imread(t).astype(np.float32) / 255.0
        #bgr =>rgb 순서로 변경
        im = im[..., ::-1]
        
        # 이미지경로에서 /img/를 /wc/로 교체체
        t = t.replace('/img/', '/wc/')
        # 마지막 세개문자를 삭제하고 exr추가, 확장자변경
        t = t[:-3] + 'exr'
        # 바꾼 경로의 exr형식의 이미지파일을 불러온다 
        wc = cv2.imread(t, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_UNCHANGED).astype(np.float32)

        t = t.replace('/wc/', '/bm/')
        t = t[:-3] + 'mat'
        # 경로의 mat 파일을 불러와서 bm키에 해당하는 데이터를 가져온다
        bm = loadmat(t)['bm']

        # 배경이미지 랜덤으로 선택해서 불러오기
        ind = random.randint(0, len(self.bgtex) - 1)
        #정규화하면서 읽기
        bg = cv2.imread(self.bgtex[ind]).astype(np.float32) / 255.0
        #200*200으로 리사이즈한다
        bg = cv2.resize(bg, (200, 200))
        # 3x3 으로이미지를 반복시킨다 =>크기가 3배가 된다
        bg = np.tile(bg, (3, 3, 1))
        
        
        # 토치텐서로 변환하면서
        # 이미지의 세로, 가로, 채널 순서로 차원을 변경
        #이미지
        im = torch.from_numpy(im.transpose((2, 0, 1)).copy())
        #깊이
        wc = torch.from_numpy(wc.transpose((2, 0, 1)).copy())
        # mat파일, 변환행렬
        bm = torch.from_numpy(bm.transpose((2, 0, 1)).copy())
        #배경
        bg = torch.from_numpy(bg.transpose((2, 0, 1)).copy())

        return im, wc, bm, bg


# 데이터를 증강한다
class Doc3DDataAug(nn.Module):
    def __init__(self):
        super(Doc3DDataAug, self).__init__()
        #  cj 객체=>  이미지의 밝기, 대비, 채도, 색조를 무작위로 조정
        self.cj = KA.ColorJitter(0.1, 0.1, 0.1, 0.1)
    
    def forward(self, img, wc, bm, bg):
        # 마스크 가져오기, 
        # 깊이 정보 wc를 이용하여 깊이가 0이 아닌 부분을 선택
        mask = (wc[:, 0] != 0) & (wc[:, 1] != 0) & (wc[:, 2] != 0)
        
        B = img.size(0)  # 이미지 배치크기를 가져온다
        # 각 이미지에 대한 무작위 패딩 및 좌표 이동을 정의하는 무작위 값 c 생성
        # 0~19 사이정수, 배치크기 B마다 5개씩 생성
        c = torch.randint(20, (B, 5))
        # 이미지 리스트
        img_list = []
        # 변환행렬 리스트
        bm_list = []
        for ii in range(B):
            x_img = img[ii]
            x_bm = bm[ii]
            x_msk = mask[ii]
            
            #마스크이미지에서 픽셀값이 0가 아닌 픽셀들의 좌표를 가져온다
            y, x = x_msk.nonzero(as_tuple=True)
            minx = x.min()
            maxx = x.max()
            miny = y.min()
            maxy = y.max()
            # 객체가 마스크된 부분을 제외한 나머지를 자른다
            x_img = x_img[:, miny : maxy + 1, minx : maxx + 1]
            x_msk = x_msk[None, miny : maxy + 1, minx : maxx + 1]

            # padding
            # 이미지와 마스크를 랜덤값으로 패딩을넣어 크기를 조절
            x_img = F.pad(x_img, c[ii, : 4].tolist())
            x_msk = F.pad(x_msk, c[ii, : 4].tolist())
            
            # 변환행렬의 좌표범위도 이미지에 맞게 조정
            x_bm[0, :, :] = (x_bm[0, :, :] - minx + c[ii][0]) / x_img.size(2) * 2 - 1
            x_bm[1, :, :] = (x_bm[1, :, :] - miny + c[ii][2]) / x_img.size(1) * 2 - 1

            # 배경이미지 추가하거나, 교체
            if c[ii][-1] > 2:
                x_bg = bg[ii][:, :x_img.size(1), :x_img.size(2)]
            else:
                #무작위 노이즈 배경, 이미지의 크기와 동일
                x_bg = torch.ones_like(x_img) * torch.rand((3, 1, 1), device=x_img.device)
            x_msk = x_msk.float()
            # 마스크를 사용해서 객체부분은 원래이미지를 사용하고 
            # 배경부분만 배경이미지로 대체한다
            x_img = x_img * x_msk + x_bg * (1. - x_msk)

            # resize
            # 256,256크기 이미지로 조절하고 배치형태로 조정
            x_img = KG.resize(x_img[None, :], (256, 256))
            #크기조정된 이미지를  리스트에 담기
            img_list.append(x_img)
            # 위이미지와 짝인 변환행렬도 저장
            bm_list.append(x_bm)
        # 합쳐서 하나의 모든 이미지를 가진 텐서로 결합
        img = torch.cat(img_list)
        # 모든 이미지의 변환행렬을 포함한 텐서로 쌓는다
        bm = torch.stack(bm_list)
        # 이미지 색상변환 적용
        img = self.cj(img)
        # 증강된 데이터들 반환
        return img, bm


# 직접실행할때, 다른곳에서 import하는게 아닐때
if __name__ == '__main__':
    # 데이터셋 인스턴스 초기화
    dt = Doc3D()
    from visdom import Visdom
    vis = Visdom(port=10086)
    x, xt, y, yt, t = dt[999]

    vis.image(x.clamp(0, 1), opts={'caption': 'x'}, win='x')
    vis.image(xt.clamp(0, 1), opts={'caption': 'xt'}, win='xt')
    vis.image(y.clamp(0, 1), opts={'caption': 'y'}, win='y')
    vis.image(yt.clamp(0, 1), opts={'caption': 'yt'}, win='yt')
    vis.image(t, opts={'caption': 't'}, win='t')
