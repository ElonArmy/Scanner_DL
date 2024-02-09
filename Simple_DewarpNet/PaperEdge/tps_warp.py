import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class TpsWarp(nn.Module):
    '''
    Thin Plate Spline (TPS) 워핑을 수행할 모듈이다
    '''
    def __init__(self, s):
        super(TpsWarp, self).__init__()
        #입력되는 크기 s에 따라 2d 좌표그리드를 생성한다
        iy, ix = torch.meshgrid(torch.linspace(-1, 1, s), torch.linspace(-1, 1, s))
        # 2d좌표그리드(ix,iy)를 x,y좌표를 가지는 2d그리드로 결합하고 1차원으로 텐서를 펼친다
        self.gs = torch.stack((ix, iy), dim=2).reshape((1, -1, 2)).to(device)
        self.sz = s  # 입력되는 크기
        
    # Tps 순전파함수
    def forward(self, src, dst):
        '''
        src: 입력 좌표그리드
        dst: 목표 좌표그리드
        (B.n.2): 배치크기, 점의개수, (x,y좌표계 차원) 
        '''
        # src and dst are B.n.2
        # B,n만 가져온다
        B, n, _ = src.size()   
        
        # B.n.1.2
        # 형태를 (B, n, 1, 2)로 바꾸기, 입력좌표그리드와 목표사이의 거리를 계산하는데 사용
        delta = src.unsqueeze(2)
        # 자기자신과의 차이를 계산해서 최동 델타(변화량)를 얻는다
        delta = delta - delta.permute(0, 2, 1, 3)
        # B.n.n
        #  delta의 2D 벡터들 간의 유클리드 거리를 계산해서 K텐서를 만든다
        K = delta.norm(dim=3)
        
        # (B, n, 3)의 형태가됨, 1차원 벡터와 원본 좌표 (x, y)를 포함함
        P = torch.cat((torch.ones((B, n, 1), device=device), src), 2)
        
        # K와 P 텐서를 연결하여 B, n, n+3)의 형태로 만듦
        L = torch.cat((K, P), 2)
        
        # P 행렬을 전치하여 (B, 3, n) 형태로 만들고
        # 원래 크기인 (B, 3, 3)으로 확장한다. 경계 조건에 사용
        t = torch.cat((P.permute(0, 2, 1), torch.zeros((B, 3, 3), device=device)), 2)
        
        # L과 t 행렬을 연결해서 최종 L 초기화, 
        # (B, n+n+3, n+n+3) 형태임,  TPS 워프 연산에 필요한 정보를 모두 가지고 있음
        L = torch.cat((L, t), 1)
        
        # the above implementation has stability problem near the boundaries
        # 변환 행렬 wv를 만들었다, 입력좌표를 목표 좌표로 변환하는데 사용
        wv = torch.solve(torch.cat((dst, torch.zeros((B, 3, 2), device=device)), 1), L)[0]

        # get the grid sampler
        s = self.gs.size(1)   # 원본 이미지의 크기 s 추출
        gs = self.gs  # 위에서 2d좌표그리드를 입력크기에 따라 1차원으로 펼친것
        delta = gs.unsqueeze(2)     #(B, s, s, 1)의 형태로 변환
        #델타에서 입력 이미지의 형태를 변환하고 목표 좌표그리드와 차이를 계산
        # (B, s, s, n, 2) 형태가 됨
        delta = delta - src.unsqueeze(1)
        
        # 2d 벡터간의 유클리드거리계산 (B, s, s, n) 형태가됨 
        K = delta.norm(dim=3)
        gs = gs.expand(B, -1, -1) # 배치크기 B에맞게 gs좌표그리드(1차원)를 확장한다
        # 상수 1과 확장된 gs를 연결하여 행렬 P를 생성, (B, s, s, 3)의 형태
        P = torch.cat((torch.ones((B, s, 1), device=device), gs), 2)
        # 유행렬 K와 P 행렬을 연결, (B, s, s, n+3)의 형태
        L = torch.cat((K, P), 2)
        # L과 wv를 행렬곱셈해서 변환된 좌표그리드 gs로 만듦
        gs = torch.matmul(L, wv)
        # gs를 원래 이미지 크기로 변환하고, 형태를 (B, 2, self.sz, self.sz)로 바꿔서 반환
        return gs.reshape(B, self.sz, self.sz, 2).permute(0, 3, 1, 2)

class PspWarp(nn.Module):
    def __init__(self):
        super().__init__()

    def pspmat(self, src, dst):
        # B, 4, 2
        B, _, _ = src.size()
        s = torch.cat([
            torch.cat([src, torch.ones((B, 4, 1), device=device), torch.zeros((B, 4, 3), device=device), 
                        -dst[..., 0 : 1] * src[..., 0 : 1], -dst[..., 0 : 1] * src[..., 1 : 2]], dim=2),
            torch.cat([torch.zeros((B, 4, 3), device=device), src, torch.ones((B, 4, 1), device=device), 
                        -dst[..., 1 : 2] * src[..., 0 : 1], -dst[..., 1 : 2] * src[..., 1 : 2]], dim=2)
        ], dim=1)
        t = torch.cat([dst[..., 0 : 1], dst[..., 1 : 2]], dim=1)
        # M = s.inverse() @ t
        M = torch.solve(t, s)[0]
        # M is B 8 1
        return M
    
    def forward(self, xy, M):
        # permute M to B 1 8
        M = M.permute(0, 2, 1)
        t = M[..., 6] * xy[..., 0] + M[..., 7] * xy[..., 1] + 1
        u = (M[..., 0] * xy[..., 0] + M[..., 1] * xy[..., 1] + M[..., 2]) / t
        v = (M[..., 3] * xy[..., 0] + M[..., 4] * xy[..., 1] + M[..., 5]) / t
        return torch.stack((u, v), dim=2)


class IdwWarp(nn.Module):
    # inverse distance weighting
    def __init__(self, s):
        super().__init__()
        iy, ix = torch.meshgrid(torch.linspace(-1, 1, s), torch.linspace(-1, 1, s))
        self.gs = torch.stack((ix, iy), dim=2).reshape((1, -1, 2)).to(device)
        self.s = s

    def forward(self, src, dst):
        # B n 2
        B, n, _ = src.size()
        # B.n.1.2
        delta = src.unsqueeze(2)
        delta = delta - self.gs.unsqueeze(0)
        # B.n.K
        p = 1
        Rsq = torch.sum(delta**2, dim=3)**p
        w = 1 / Rsq
        # turn inf to [0...1...0]
        t = torch.isinf(w)
        idx = t.any(dim=1).nonzero()
        w[idx[:, 0], :, idx[:, 1]] = t[idx[:, 0], :, idx[:, 1]].float()
        wwx = w * dst[..., 0 : 1]
        wwx = wwx.sum(dim=1) / w.sum(dim=1)
        wwy = w * dst[..., 1 : 2]
        wwy = wwy.sum(dim=1) / w.sum(dim=1)
        gs = torch.stack((wwx, wwy), dim=2).reshape(B, self.s, self.s, 2).permute(0, 3, 1, 2)
        return gs


if __name__ == "__main__":
    import cv2
    import numpy as np
    from hdf5storage import loadmat
    from visdom import Visdom
    
    # localhost:10086 서버에서 확인
    vis = Visdom(port=10086)

    tpswarp = TpsWarp(16)
    import matplotlib.pyplot as plt
    cn = torch.tensor([[-1, -1], [1, -1], [1, 1], [-1, 1], [-0.5, -1], [0, -1], [0.5, -1]], dtype=torch.float, device=device).unsqueeze(0)
    pn = torch.tensor([[-1, -0.5], [1, -1], [1, 1], [-1, 0.5], [-0.5, -1], [0, -0.5], [0.5, -1]], device=device).unsqueeze(0)
    pspwarp = PspWarp()

    M = pspwarp.pspmat(cn[..., 0 : 4, :], pn[..., 0 : 4, :])
    invM = pspwarp.pspmat(pn[..., 0 : 4, :], cn[..., 0 : 4, :])
   

    t = tpswarp(cn, pn)
    from tsdeform import WarperUtil
    wu = WarperUtil(16)
    tgs = wu.global_post_warp(t, 16, invM, M)

    
    t = tgs.permute(0, 2, 3, 1)[0].detach().cpu().numpy()

    plt.clf()
    plt.pcolormesh(t[..., 0], t[..., 1], np.zeros_like(t[..., 0]), edgecolors='r')
    plt.gca().invert_yaxis()
    plt.gca().axis('equal')
    vis.matplot(plt, env='grid', win='mpl')

