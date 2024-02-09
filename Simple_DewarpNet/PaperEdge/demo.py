# -*- encoding: utf-8 -*-
import argparse
import copy
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from networks import GlobalWarper, LocalWarper, WarperUtil

cv2.setNumThreads(0)   # cpu스레드 설정 
cv2.ocl.setUseOpenCL(False)  #opencl 비활성


# 입력 이미지 변환
def load_img(img_path):
    #이미지를 float32로 픽셀값을 0~1로 정규화 해서 cv로 읽는다
    im = cv2.imread(img_path).astype(np.float32) / 255.0
    # 채널을 순서를 RGB순으로 변경
    im = im[:, :, (2, 1, 0)]
    # 256*256 사이즈로 변경
    # cv2.INTER_AREA: 크기를 줄일때 주로사용하는 평균보간 리사이징 알고리즘
    im = cv2.resize(im, (256, 256), interpolation=cv2.INTER_AREA)
    # 배열의 차원순서 (높이, 너비, 채널)에서 (채널, 높이, 너비)로 변경하고
    # 넘파이배열을 파이토치텐서로 변환
    im = torch.from_numpy(np.transpose(im, (2, 0, 1)))
    return im


if __name__ == '__main__':
    # 커맨드라인에서 들어오는 인수 파싱하는거
    parser = argparse.ArgumentParser()
    # 모델경로, 이미지 경로, 출력저장 경로
    parser.add_argument('--Enet_ckpt', type=str,
                        default='models/G_w_checkpoint_13820.pt')
    parser.add_argument('--Tnet_ckpt', type=str,
                        default='models/L_w_checkpoint_27640.pt')
    parser.add_argument('--img_path', type=str, default='images/normal.jpg')
    parser.add_argument('--out_dir', type=str, default='output')
    # 파싱한 인수들을 객체에 담아 저장
    args = parser.parse_args()

    img_path = args.img_path
    dst_dir = args.out_dir
    # 없으면 생성, 있으면 경로에 사용
    Path(dst_dir).mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 네트워크 불러오기
    netG = GlobalWarper().to(device)
    # 네트워트에 사전학습된 가중치를 넣는다
    netG.load_state_dict(torch.load(args.Enet_ckpt, map_location=torch.device(device))['G'])
    
    ##### 이 부분에서 netG 파인튜닝 ####
    
    netG.eval() #검증모드로 변경

    netL = LocalWarper().to(device)
    netL.load_state_dict(torch.load(args.Tnet_ckpt, map_location=torch.device(device))['L'])
    
    ##### 이 부분에서 netL 파인튜닝 ####
   
    netL.eval()

    warpUtil = WarperUtil(64).to(device)

    # 네트워크가 변형 그리드를 만든다(예측)
    # 변형그리드에는 이미지의 픽셀이 어디로 어떻게 이동하면될지에 관한 정보가 들어있다.
    # grid_sample() 사용해 이미지에 변형그리드를 적용할수있다.
    # 글로벌 변형그리드, 로컬 변형그리드이다
    gs_d, ls_d = None, None
    # 그래이디언트계산 없이 실행
    with torch.no_grad():
        x = load_img(img_path)  # 이미지 불러오기
        x = x.unsqueeze(0)   # 배치형태로 변환, 단일이미지를 예측할때 사용
        x = x.to(device)
        # global 변형 그리드를 예측한다.
        d = netG(x)  # d_E the edged-based deformation field
        
        # global 변형 그리드 후처리
        # d는 변형된 이미지의 픽셀이 원본 이미지 대비 어떤 위치로 
        # 이동해야 하는지를 나타내는 좌표 정보를 담고 있는 텐서임
        d = warpUtil.global_post_warp(d, 64)
        # 객체정보를 모두가져와 완전히 새로운 객체를 만드는 복사, 영향을 주지않음
        gs_d = copy.deepcopy(d)  #=> 원본이미지를 글로벌 워핑하는것에사용
        
        # 글로벌 변형 그리드를 256*256 사이즈로 보간한다
        d = F.interpolate(d, size=256, mode='bilinear', align_corners=True)
        # 글로벌 변형그리드로 원본 이미지텐서를 변형한다
        y0 = F.grid_sample(x, d.permute(0, 2, 3, 1), align_corners=True)
        # 글로벌 워핑된 텐서를 Tnet에 넣어 로컬 변형 그리드를 만든다
        ls_d = netL(y0)
        # 문서내부의 골곡을 어떻게 펼치면되는지에 대한 정보를 
        # 가지고있는 변형 그리드이다
        # 로컬 변형 그리드의 사이즈 크기를 256로 보간한다
        ls_d = F.interpolate(ls_d, size=256, mode='bilinear', align_corners=True)
        # 변형그리드내의 픽셀값? 텐서값을 min값과 max값을 고정해서 안정화한다
        ls_d = ls_d.clamp(-1.0, 1.0)
        
    # 다시 입력 이미지를 불러와서
    # 모델에서 예측된 변형그리드를 가져와 적용한다
    im = cv2.imread(img_path).astype(np.float32) / 255.0
    # 이미지의 차원을 (높이, 너비, 채널)에서 (채널, 높이, 너비)로
    # 변경 후 토치텐서로 변환
    im = torch.from_numpy(np.transpose(im, (2, 0, 1)))
    # 텐서의 차원을 늘리고 디바이스 전달
    im = im.to(device).unsqueeze(0)  # 단일이미지
    
    # 변형 그리드의 크기를 원본 im의 높이와 너비에 맞춰 보간한다
    gs_d = F.interpolate(gs_d, (im.size(2), im.size(3)), mode='bilinear', align_corners=True)
    # gs_d의 차원을 0231순으로 재배치하고
    # 이미지에 변형 그리드를 적용한다  => 글로벌 워핑완료
    gs_y = F.grid_sample(im, gs_d.permute(0, 2, 3, 1), align_corners=True).detach()
    
    # 즉 gs_y는 문서의 경계선을 잡고 펼친 이미지의 텐서인것이다
    # 차원을 줄이고, 차원순서를 바꾸고 넘파이배열로 변경한다
    tmp_y = gs_y.squeeze().permute(1, 2, 0).cpu().numpy()
    # Enet(global deformation) 결과물 출력 
    cv2.imwrite(f'{dst_dir}/result_gs.png', tmp_y * 255.)

    # 로컬 변형그리드를 입력이미지의 크기에 맞춰 보간한다
    ls_d = F.interpolate(ls_d, (im.size(2), im.size(3)), mode='bilinear', align_corners=True)
    # 이미 글로벌 변형그리드로 변형된 gs_y 이미지를 가져와 로컬 변형그리드를 적용해 변환한다.
    ls_y = F.grid_sample(gs_y, ls_d.permute(0, 2, 3, 1), align_corners=True).detach()
    # 차원을 줄이고, 차원순서를 바꾸고 넘파이배열로 변경한다
    # 이것이 최종 디워핑된 문서의 결과물이다.
    ls_y = ls_y.squeeze().permute(1, 2, 0).cpu().numpy()
    # 최종 Tnet (local deformation) 결과물을 출력한다
    cv2.imwrite(f'{dst_dir}/result_ls.png', ls_y * 255.)
