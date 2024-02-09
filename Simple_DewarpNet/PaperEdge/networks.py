import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
import random
import numpy as np
import cv2

torch.autograd.set_detect_anomaly(True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.actv = nn.ReLU()
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride


    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.actv(out)
        
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)
            
        out += identity
        out = self.actv(out)

        return out

def _make_layer(block, inplanes, planes, blocks, stride=1, dilate=False):
        norm_layer = nn.BatchNorm2d
        downsample = None
        
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion, 1, stride, bias=False),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample, norm_layer=norm_layer))
        
        for _ in range(1, blocks):
            layers.append(block(planes, planes, 
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

class Interpolate(nn.Module):
    def __init__(self, size, mode):
        super(Interpolate, self).__init__()
        self.interp = nn.functional.interpolate
        self.size = size
        self.mode = mode
        
    def forward(self, x):
        x = self.interp(x, size=self.size, mode=self.mode, align_corners=True)
        return x
    
class GlobalWarper(nn.Module):
    def __init__(self):
        super(GlobalWarper, self).__init__()
        
        modules = [
            nn.Conv2d(5, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        ]

        planes = [64, 128, 256, 256, 512, 512]
        strides = [2, 2, 2, 2, 2]
        blocks = [1, 1, 1, 1, 1]
        
        for k in range(len(planes) - 1):
            modules.append(_make_layer(BasicBlock, planes[k], planes[k + 1], blocks[k], strides[k]))
            
        self.encoder = nn.Sequential(*modules)

        modules = []
        planes = [512, 512, 256, 128, 64]
        strides = [2, 2, 2, 2]
        blocks = [1, 1, 1, 1]
        for k in range(len(planes) - 1):
            modules += [nn.Sequential(nn.Upsample(scale_factor=strides[k], mode='bilinear', align_corners=True), 
                        _make_layer(BasicBlock, planes[k], planes[k + 1], blocks[k], 1))]
            
        self.decoder = nn.Sequential(*modules)
        
        self.to_warp = nn.Sequential(nn.Conv2d(64, 2, 1))
        self.to_warp[0].weight.data.fill_(0.0)
        self.to_warp[0].bias.data.fill_(0.0)

        
        iy, ix = torch.meshgrid(torch.linspace(-1, 1, 256), torch.linspace(-1, 1, 256))
        
        self.coord = torch.stack((ix, iy), dim=0).unsqueeze(0).to(device)
        
        iy, ix = torch.meshgrid(torch.linspace(-1, 1, 64), torch.linspace(-1, 1, 64))
        self.basegrid = torch.stack((ix * 0.9, iy * 0.9), dim=0).unsqueeze(0).to(device)
        
    def forward(self, im):
        
        B = im.size(0)
        
        c = self.coord.expand(B, -1, -1, -1).detach()
        t = torch.cat((im, c), dim=1)
        
        t = self.encoder(t)
        t = self.decoder(t)
        t = self.to_warp(t)
        
        gs = t + self.basegrid

        return gs
    
class LocalWarper(nn.Module):

    def __init__(self):
        super().__init__()
        modules = [
            nn.Conv2d(5, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        ]
        
        planes = [64, 128, 256, 256, 512, 512]
        strides = [2, 2, 2, 2, 2]
        blocks = [1, 1, 1, 1, 1]
        for k in range(len(planes) - 1):
            modules.append(_make_layer(BasicBlock, planes[k], planes[k + 1], blocks[k], strides[k]))
        self.encoder = nn.Sequential(*modules)

        modules = []
        planes = [512, 512, 256, 128, 64]
        strides = [2, 2, 2, 2]
        blocks = [1, 1, 1, 1]
        for k in range(len(planes) - 1):
            modules += [nn.Sequential(nn.Upsample(scale_factor=strides[k], mode='bilinear', align_corners=True), 
                        _make_layer(BasicBlock, planes[k], planes[k + 1], blocks[k], 1))]
        self.decoder = nn.Sequential(*modules)

        self.to_warp = nn.Sequential(nn.Conv2d(64, 2, 1))
        self.to_warp[0].weight.data.fill_(0.0)
        self.to_warp[0].bias.data.fill_(0.0)

        iy, ix = torch.meshgrid(torch.linspace(-1, 1, 256), torch.linspace(-1, 1, 256))
        self.coord = torch.stack((ix, iy), dim=0).unsqueeze(0).to(device)
        iy, ix = torch.meshgrid(torch.linspace(-1, 1, 64), torch.linspace(-1, 1, 64))
        self.basegrid = torch.stack((ix, iy), dim=0).unsqueeze(0).to(device)


    def forward(self, im):
        c = self.coord.expand(im.size(0), -1, -1, -1).detach()
        t = torch.cat((im, c), dim=1)

        t = self.encoder(t)
        t = self.decoder(t)
        t = self.to_warp(t)

        t[..., 1, 0, :] = 0
        t[..., 1, -1, :] = 0
        t[..., 0, :, 0] = 0
        t[..., 0, :, -1] = 0

        gs = t + self.basegrid
        return gs


def gs_to_bd(gs):
    t = torch.cat([gs[..., 0, :], gs[..., -1, :], gs[..., 1 : -1, 0], gs[..., 1 : -1, -1]], dim=2).permute(0, 2, 1)
    return t

from tps_warp import TpsWarp, PspWarp

class MaskLoss(nn.Module):

    def __init__(self, gsize):
        super().__init__()
        self.tpswarper = TpsWarp(gsize)
        self.pspwarper = PspWarp()
        self.msk = torch.ones(1, 1, gsize, gsize, device=device)
        self.cn = torch.tensor([[-1, -1], [1, -1], [1, 1], [-1, 1]], dtype=torch.float, device=device).unsqueeze(0)

    def forward(self, gs, y, s):
        B, _, s0, _ = gs.size()
        tgs = F.interpolate(gs, s, mode='bilinear', align_corners=True)

        srcpts = gs_to_bd(tgs)
        iy, ix = torch.meshgrid(torch.linspace(-1, 1, s), torch.linspace(-1, 1, s))
        t = torch.stack((ix, iy), dim=0).unsqueeze(0).to(device).expand_as(tgs)
        dstpts = gs_to_bd(t)

        tgs_f = self.tpswarper(srcpts, dstpts.detach())
        ym = self.msk.expand_as(y)
        yh = F.grid_sample(ym, tgs_f.permute(0, 2, 3, 1), align_corners=True)
        loss_f = F.l1_loss(yh, y)

        tgs_b = self.tpswarper(dstpts.detach(), srcpts)
        yy = F.grid_sample(y, tgs_b.permute(0, 2, 3, 1), align_corners=True)
        loss_b = F.l1_loss(yy, ym)
        
        return loss_f + loss_b, tgs_f

    def _dist(self, x):
        x = torch.cat([x[..., 0 : 1].detach(), x[..., 1 : -1], x[..., -1 : ].detach()], dim=2)
        d = x[..., 1:] - x[..., :-1]
        return torch.norm(d, dim=1)
    
class WarperUtil(nn.Module):

    def __init__(self, imsize):
        super().__init__()
        self.tpswarper = TpsWarp(imsize)
        self.pspwarper = PspWarp()
        self.s = imsize
    
    def global_post_warp(self, gs, s):
        gs = F.interpolate(gs, s, mode='bilinear', align_corners=True)

        m1 = gs[..., 0, :]
        m2 = gs[..., -1, :]
        n1 = gs[..., 0]
        n2 = gs[..., -1]
        m1x_interval_ratio = m1[:, 0, 1:] - m1[:, 0, :-1]
        m1x_interval_ratio /= m1x_interval_ratio.sum(dim=1, keepdim=True)
        m2x_interval_ratio = m2[:, 0, 1:] - m2[:, 0, :-1]
        m2x_interval_ratio /= m2x_interval_ratio.sum(dim=1, keepdim=True)
        t = torch.stack([m1x_interval_ratio, m2x_interval_ratio], dim=1).unsqueeze(1)
        mx_interval_ratio = F.interpolate(t, (s, m1x_interval_ratio.size(1)), mode='bilinear', align_corners=True)
        mx_interval = (n2[..., 0 : 1, :] - n1[..., 0 : 1, :]).unsqueeze(3) * mx_interval_ratio
        dx = torch.cumsum(mx_interval, dim=3) + n1[..., 0 : 1, :].unsqueeze(3)
        dx = dx[..., 1 : -1, :-1]
        n1y_interval_ratio = n1[:, 1, 1:] - n1[:, 1, :-1]
        n1y_interval_ratio /= n1y_interval_ratio.sum(dim=1, keepdim=True)
        n2y_interval_ratio = n2[:, 1, 1:] - n2[:, 1, :-1]
        n2y_interval_ratio /= n2y_interval_ratio.sum(dim=1, keepdim=True)
        t = torch.stack([n1y_interval_ratio, n2y_interval_ratio], dim=2).unsqueeze(1)
        ny_interval_ratio = F.interpolate(t, (n1y_interval_ratio.size(1), s), mode='bilinear', align_corners=True)
        ny_interval = (m2[..., 1 : 2, :] - m1[..., 1 : 2, :]).unsqueeze(2) * ny_interval_ratio
        dy = torch.cumsum(ny_interval, dim=2) + m1[..., 1 : 2, :].unsqueeze(2)
        dy = dy[..., :-1, 1 : -1]
        ds = torch.cat((dx, dy), dim=1)
        gs[..., 1 : -1, 1 : -1] = ds
        return gs

    def perturb_warp(self, dd):
        B = dd.size(0)
        s = self.s
        iy, ix = torch.meshgrid(torch.linspace(-1, 1, s), torch.linspace(-1, 1, s))
        t = torch.stack((ix, iy), dim=0).unsqueeze(0).to(device).expand(B, -1, -1, -1)

        tt = t.clone()

        nd = random.randint(0, 4)
        for ii in range(nd):
            pm = (torch.rand(B, 1) - 0.5) * 0.2
            ps = (torch.rand(B, 1) - 0.5) * 1.95
            pt = ps + pm
            pt = pt.clamp(-0.975, 0.975)
            a1 = (torch.rand(B, 2) > 0.5).float() * 2 -1
            a2 = torch.rand(B, 1) > 0.5
            a2 = torch.cat([a2, a2.bitwise_not()], dim=1)
            a3 = a1.clone()
            a3[a2] = ps.view(-1)
            ps = a3.clone()
            a3[a2] = pt.view(-1)
            pt = a3.clone()
            bds = torch.stack([
                t[0, :, 1 : -1, 0], t[0, :, 1 : -1, -1], t[0, :, 0, 1 : -1], t[0, :, -1, 1 : -1]
            ], dim=2)

            pbd = a2.bitwise_not().float() * a1
            pbd = torch.abs(0.5 * pbd[:, 0] + 2.5 * pbd[:, 1] + 0.5).long()
            pbd = torch.stack([pbd + 1, pbd + 2, pbd + 3], dim=1) % 4
            pbd = bds[..., pbd].permute(2, 0, 1, 3).reshape(B, 2, -1)            

            srcpts = torch.stack([
                t[..., 0, 0], t[..., 0, -1], t[..., -1, 0], t[..., -1, -1],
                ps.to(device)
            ], dim=2)
            srcpts = torch.cat([pbd, srcpts], dim=2).permute(0, 2, 1)
            dstpts = torch.stack([
                t[..., 0, 0], t[..., 0, -1], t[..., -1, 0], t[..., -1, -1],
                pt.to(device)
            ], dim=2)
            dstpts = torch.cat([pbd, dstpts], dim=2).permute(0, 2, 1)

            tgs = self.tpswarper(srcpts, dstpts)
            tt = F.grid_sample(tt, tgs.permute(0, 2, 3, 1), align_corners=True)

        nd = random.randint(1, 5)
        for ii in range(nd):

            pm = (torch.rand(B, 2) - 0.5) * 0.2
            ps = (torch.rand(B, 2) - 0.5) * 1.95
            pt = ps + pm
            pt = pt.clamp(-0.975, 0.975)

            srcpts = torch.cat([
                t[..., -1, :], t[..., 0, :], t[..., 1 : -1, 0], t[..., 1 : -1, -1],
                ps.unsqueeze(2).to(device)
            ], dim=2).permute(0, 2, 1)
            dstpts = torch.cat([
                t[..., -1, :], t[..., 0, :], t[..., 1 : -1, 0], t[..., 1 : -1, -1],
                pt.unsqueeze(2).to(device)
            ], dim=2).permute(0, 2, 1)
            tgs = self.tpswarper(srcpts, dstpts)
            tt = F.grid_sample(tt, tgs.permute(0, 2, 3, 1), align_corners=True)
        tgs = tt

        num_sample = 512
        n = s * s
        idx = torch.randperm(n)
        idx = idx[:num_sample]
        srcpts = tgs.reshape(-1, 2, n)[..., idx].permute(0, 2, 1)
        dstpts = t.reshape(-1, 2, n)[..., idx].permute(0, 2, 1)
        invtgs = self.tpswarper(srcpts, dstpts)
        return tgs, invtgs

    def equal_spacing_interpolate(self, gs, s):
        def equal_bd(x, s):
            v0 = x[..., :-1] # B 2 n-1
            v = x[..., 1:] - x[..., :-1]
            vn = v.norm(dim=1, keepdim=True)
            v = v / vn
            c = vn.sum(dim=2, keepdim=True) #B 1 1
            a = vn / c
            b = torch.cumsum(a, dim=2)
            b = torch.cat((torch.zeros(B, 1, 1, device=device), b[..., :-1]), dim=2)
            
            t = torch.linspace(1e-5, 1 - 1e-5, s).view(1, s, 1).to(device)
            t = t - b # B s n-1
            
            tt = torch.cat((t, -torch.ones(B, s, 1, device=device)), dim=2) # B s n
            tt = tt[..., 1:] * tt[..., :-1] # B s n-1
            tt = (tt < 0).float()
            d = torch.matmul(v0, tt.permute(0, 2, 1)) + torch.matmul(v, (tt * t).permute(0, 2, 1)) # B 2 s
            return d

        gs = F.interpolate(gs, s, mode='bilinear', align_corners=True)
        B = gs.size(0)
        dst_cn = torch.tensor([[-1, -1], [1, -1], [1, 1], [-1, 1]], dtype=torch.float, device=device).expand(B, -1, -1)
        src_cn = torch.stack([gs[..., 0, 0], gs[..., 0, -1], gs[..., -1, -1], gs[..., -1, 0]], dim=2).permute(0, 2, 1)
        M = self.pspwarper.pspmat(src_cn, dst_cn).detach()
        invM = self.pspwarper.pspmat(dst_cn, src_cn).detach()
        pgs = self.pspwarper(gs.permute(0, 2, 3, 1).reshape(B, -1, 2), M).reshape(B, s, s, 2).permute(0, 3, 1, 2)
        t = [pgs[..., 0, :], pgs[..., -1, :], pgs[..., :, 0], pgs[..., :, -1]]
        d = []
        for x in t:
            d.append(equal_bd(x, s))
        pgs[..., 0, :] = d[0]
        pgs[..., -1, :] = d[1]
        pgs[..., :, 0] = d[2]
        pgs[..., :, -1] = d[3]
        gs = self.pspwarper(pgs.permute(0, 2, 3, 1).reshape(B, -1, 2), invM).reshape(B, s, s, 2).permute(0, 3, 1, 2)
        gs = self.global_post_warp(gs, s)
        return gs
    
class LocalLoss(nn.Module):

    def __init__(self):
        super().__init__()

    def identity_loss(self, gs):

        s = gs.size(2)
        iy, ix = torch.meshgrid(torch.linspace(-1, 1, s), torch.linspace(-1, 1, s))
        t = torch.stack((ix, iy), dim=0).unsqueeze(0).to(device).expand_as(gs)
        loss = F.l1_loss(gs, t.detach())
        return loss

    def direct_loss(self, gs, invtgs):

        loss = F.l1_loss(gs, invtgs.detach())
        return loss

    def warp_diff_loss(self, xd, xpd, tgs, invtgs):

        loss_f = F.l1_loss(xd, F.grid_sample(tgs, xpd.permute(0, 2, 3, 1), align_corners=True).detach())
        loss_b = F.l1_loss(xpd, F.grid_sample(invtgs, xd.permute(0, 2, 3, 1), align_corners=True).detach())
        loss = loss_f + loss_b
        return loss


class SupervisedLoss(nn.Module):

    def __init__(self):
        super().__init__()
        s = 64
        self.tpswarper = TpsWarp(s)

    def fm2bm(self, fm):
 
        B, _, s, _ = fm.size()
        iy, ix = torch.meshgrid(torch.linspace(-1, 1, s), torch.linspace(-1, 1, s))
        t = torch.stack((ix, iy), dim=0).unsqueeze(0).to(device).expand(B, -1, -1, -1)
        srcpts = []
        dstpts = []
        for ii in range(B):
            # mask
            m = fm[ii, 2]
            # z s
            z = torch.nonzero(m, as_tuple=False)
            num_sample = 512
            n = z.size(0)
            idx = torch.randperm(n)
            idx = idx[:num_sample]
            dstpts.append(t[ii, :, z[idx, 0], z[idx, 1]])
            srcpts.append(fm[ii, : 2, z[idx, 0], z[idx, 1]] * 2 - 1)
        srcpts = torch.stack(srcpts, dim=0).permute(0, 2, 1)
        dstpts = torch.stack(dstpts, dim=0).permute(0, 2, 1)

        bm = self.tpswarper(srcpts, dstpts)

        return bm
    
    def gloss(self, x, y):

        xbd = gs_to_bd(x)
        y = F.interpolate(y, 64, mode='bilinear', align_corners=True)
        
        ybd = gs_to_bd(y).detach()
        loss = F.l1_loss(xbd, ybd.detach())
        return loss

    def lloss(self, x, y, dg):

        B, _, s, _ = dg.size()
        iy, ix = torch.meshgrid(torch.linspace(-1, 1, s), torch.linspace(-1, 1, s))
        t = torch.stack((ix, iy), dim=0).unsqueeze(0).to(device).expand(B, -1, -1, -1)
        num_sample = 512

        n = s * s
        idx = torch.randperm(n)
        idx = idx[:num_sample]
        srcpts = dg.reshape(-1, 2, n)[..., idx].permute(0, 2, 1)
        dstpts = t.reshape(-1, 2, n)[..., idx].permute(0, 2, 1)
        invdg = self.tpswarper(srcpts, dstpts)
        dl = F.grid_sample(invdg, y.permute(0, 2, 3, 1), align_corners=True)
        dl = F.interpolate(dl, 64, mode='bilinear', align_corners=True)
        loss = F.l1_loss(x, dl.detach())
        return loss, dl.detach()