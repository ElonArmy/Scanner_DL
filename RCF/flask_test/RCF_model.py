import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import time
import cv2
import numpy as np
import datetime
import matplotlib.pyplot as plt
import imutils
from imutils.perspective import four_point_transform


# 다른 모듈에서 import * 할 때 가져올것들 정의
# __all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']


# 사용가능한 여러버전의 resnet 사전학습모델들. 버전마다 속도, 성능등 특징이다르다
model_urls = {
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
}


# 3x3 컨볼루션 레이어정의 => 이렇게 사용하는것을 헬퍼 함수라한다
def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

# ResNet 아키텍처에서 사용되는 기본 요소 클래스
class BasicBlock(nn.Module):
    #ResNet에서 사용되는 확장의 비율
    #출력 피쳐와 입력 피쳐가 같기때문에 1이다
    expansion = 1
    
    #레이어 정의
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        
    # 네트워크 연결 정의(순전파 함수)
    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        
        # 선택적으로 다운샘플링도 한다
        # 입력과 출력의 차원을 맞추기위한 용도
        if self.downsample is not None:
            identity = self.downsample(x)
            
        # shortcut connection이라 한다
        # 입력에 출력을 더한다 =>잔차를 구하는 것이다
        # 네트워크가 학습하는것은 잔차이다 => 이런 모델학습을 residual learning 이라한다
        # 네트워크가 깊어질때 vanishing gradient 문제를 완화
        out += identity
        out = self.relu(out)

        return out


# ResNet에서 네트워크가 깊어져도 파라미터를 효율적으로 관리하기위한 클래스
# 병목현상 => 피쳐의 차원축소 또는 차원확장을 할때 파라미터(자동차) 관리를 효율적으로 하는 역할
class Bottleneck(nn.Module):
    # 출력 피쳐 = 입력피쳐 x 4 
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        #1x1conv는 입력피쳐의 차원을 축소한다 
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        # 출력 채널수에 곱해져 차원을 확장한다.
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

# resnet 모델 : 다양한 신경망 레이어를 통과할 때 이미지의 여러특성을 감지한다.
# 감지한 특성을 통합하여 최종 출력을 생성하는 구조
class ResNet(nn.Module):
    
    # block: 사용할 블록, layers: 각 레이어에 있는 블록의 개수 리스트
    # zero_init_residual: 잔차블록을 마지막에 배치하여 정규화레이어를 0으로 초기화할지 여부
    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # layers 리스트로 들어오는대로 각각 정의됨
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        # 평균풀링 떄려서 1크기로 특성맵을 줄임
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # 각 resnet 레이어에서 출력을 다운 샘플링함
        self.C1_down_channel = nn.Conv2d(64, 21, 1)
        self.C2_down_channel = nn.Conv2d(256, 21, 1)
        self.C3_down_channel = nn.Conv2d(512, 21, 1)
        self.C4_down_channel = nn.Conv2d(1024, 21, 1)
        self.C5_down_channel = nn.Conv2d(2048, 21, 1)

        # 다운샘플링된 출력에서 예측 점수를 계산함
        self.score_dsn1 = nn.Conv2d(21, 1, 1)
        self.score_dsn2 = nn.Conv2d(21, 1, 1)
        self.score_dsn3 = nn.Conv2d(21, 1, 1)
        self.score_dsn4 = nn.Conv2d(21, 1, 1)
        self.score_dsn5 = nn.Conv2d(21, 1, 1)
        self.score_final = nn.Conv2d(5, 1, 1)

        # 가중치 초기화
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # He 초기화로 conv레이어 가중치 초기화 relu와 함께사용
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                # 배치정규화레이어의 가중치를 1, 편향을 0으로 초기화
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            # 이 옵션이 활성화되면, 잔차 블록의 마지막 배치 정규화 레이어의 가중치를 0으로 초기화합니다.
            # 이는 잔차 블록이 초기에 항등 매핑과 비슷한 역할을 하도록 만들어 
            # 네트워크의 학습을 안정화시킬 수 있습니다.
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    # 여러개의 블록, 보틀넥을 연결해서 하나의 레이어를 만드는 함수
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            # 입출력 차원이 다를때 맞춰주는 다운 샘플링
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x, size):
        #x 1
        x = self.conv1(x)  #1/2
        x = self.bn1(x)
        x = self.relu(x)
        C1 = self.maxpool(x) #1/4
        C2 = self.layer1(C1) #1/4
        C3 = self.layer2(C2) #1/8
        C4 = self.layer3(C3) #1/16
        C5 = self.layer4(C4) #1/32


        R1 = self.relu(self.C1_down_channel(C1))
        R2 = self.relu(self.C2_down_channel(C2))
        R3 = self.relu(self.C3_down_channel(C3))
        R4 = self.relu(self.C4_down_channel(C4))
        R5 = self.relu(self.C5_down_channel(C5))

        so1_out = self.score_dsn1(R1)
        so2_out = self.score_dsn2(R2)
        so3_out = self.score_dsn3(R3)
        so4_out = self.score_dsn4(R4)
        so5_out = self.score_dsn4(R5)

        upsample = nn.UpsamplingBilinear2d(size)

        out1 = upsample(so1_out)
        out2 = upsample(so2_out)
        out3 = upsample(so3_out)
        out4 = upsample(so4_out)
        out5 = upsample(so5_out)

        fuse = torch.cat([out1, out2, out3, out4, out5], dim=1)
        final_out = self.score_final(fuse)

        results = [out1, out2, out3, out4, out5, final_out]
        results = [torch.sigmoid(r) for r in results]
        return results
    
def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']), strict=False)
    return model


# pre-trained model 다운로드 https://drive.google.com/file/d/15dl2GkxsRuy6ovWYZ8uGiYCgNwXAhcNj/view
PATH_WEIGHT = '../only-final-lr-0.01-iter-130000.pth'

# gpu또는 cpu사용
device = ('cuda' if torch.cuda.is_available()
          else 'cpu')

# 모델 추론클래스 생성
# resnet 기반의 RCF로 edge detection 수행
class RCF():
    def __init__(self, device=device):
        
        tstamp = time.time()
        self.device = device

        device = torch.device(device)
        # resnet101 학습되지않은 모델을 불러온다
        self.net = resnet101(pretrained=False)
        print('[RCF] loading with', self.device)
        # 다운받은 사전학습된 가중치를 모델 내부사전에 넣어준다 dict형태이다
        self.net.load_state_dict(torch.load(PATH_WEIGHT, map_location=device))
        
        # 평가모드 실행 => 드롭아웃, 배치놈 등을 추론모드로해서 예측가능하게함
        self.net.eval()
        print('[RCF] finished loading (%.4f sec)' % (time.time() - tstamp))

    # 경계선 감지 함수
    def detect_edge(self, img):
        start_time = datetime.datetime.now()
        print('시작시간 : {}'.format(start_time))
        
        #입력 이미지 전처리
        org_img = np.array(img, dtype=np.float32)
        h, w, _ = org_img.shape

        pre_img = self.prepare_image_cv2(org_img)
        pre_img = torch.from_numpy(pre_img).unsqueeze(0)
        
        # 모델에 넣어서 이미지의 각 픽셀의 경계를 출력
        outs = self.net(pre_img, (h, w))
        # 경계선을 명확하게하고 최종경계선 이미지 생성
        result = outs[-1].squeeze().detach().cpu().numpy()

        result = (result * 255).astype(np.uint8)

        end_time = datetime.datetime.now()
        print('종료시간 : {}'.format(end_time))

        time_delta = end_time - start_time
        print('수행시간 : {} 초'.format(time_delta.seconds) + "\n")

        return result
    
    # 이미지 전처리 함수
    # 이미지 사이즈를 조정하고 
    # 이미지 색 채널(차원) 순서를 바꾼다 BGR => RGB 순으로 변환
    def prepare_image_cv2(self, im):
        # im -= np.array((104.00698793,116.66876762,122.67891434))
        im = cv2.resize(im, dsize=(1024, 1024), interpolation=cv2.INTER_LINEAR)
        im = np.transpose(im, (2, 0, 1))  # (H x W x C) to (C x H x W)

        return im


# utils: 필요한 여러기능을 함수로 만드러 모듈에 모아두었따
# 모아 두어야하는데 패키지 의존성때문에 model에 같이 넣었따


def prepare_image_PIL(im):
    im = im[:,:,::-1] - np.zeros_like(im) # rgb to bgr
    # im -= np.array((104.00698793,116.66876762,122.67891434))
    im = np.transpose(im, (2, 0, 1)) # (H x W x C) to (C x H x W)
    return im

def prepare_image_cv2(im):
    # im -= np.array((104.00698793,116.66876762,122.67891434))
    im = cv2.resize(im, dsize=(1024, 1024), interpolation=cv2.INTER_LINEAR)
    im = np.transpose(im, (2, 0, 1)) # (H x W x C) to (C x H x W)
    return im

def plt_imshow(title='image', img=None, figsize=(8, 5)):
    plt.figure(figsize=figsize)

    if type(img) == list:
        if type(title) == list:
            titles = title
        else:
            titles = []

            for i in range(len(img)):
                titles.append(title)

        for i in range(len(img)):
            if len(img[i].shape) <= 2:
                rgbImg = cv2.cvtColor(img[i], cv2.COLOR_GRAY2RGB)
            else:
                rgbImg = cv2.cvtColor(img[i], cv2.COLOR_BGR2RGB)

            plt.subplot(1, len(img), i + 1), plt.imshow(rgbImg)
            plt.title(titles[i])
            plt.xticks([]), plt.yticks([])

        plt.show()
    else:
        if len(img.shape) < 3:
            rgbImg = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            rgbImg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        plt.imshow(rgbImg)
        plt.title(title)
        plt.xticks([]), plt.yticks([])
        plt.show()


# 꼭짓점을 찾는 함수
def find_contours(img, thresh=-1):
    # contours를 찾아 크기순으로 정렬
    if thresh > -1:
        ret, edge_img = cv2.threshold(img, thresh, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    else:
        edge_img = img.copy()

    cnts = cv2.findContours(edge_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

    receiptCnt = None

    # 정렬된 contours를 반복문으로 수행하며 4개의 꼭지점을 갖는 도형을 검출
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        # 가장큰 사각형 영역을 찾는다
        # contours가 크기순으로 정렬되어 있기때문에 제일 첫번째 사각형을 문서의 영역으로 판단하고 break
        if len(approx) == 4:
            receiptCnt = approx
            break

    # 만약 추출한 윤곽이 없을 경우 오류
    if receiptCnt is None:
        raise Exception(("Could not find receipt outline."))

    return receiptCnt

# 꼭짓점을 연결하여 그려주는 함수
def draw_contours(img, contour):
    output = img.copy()
    cv2.drawContours(output, [contour], -1, (0, 255, 255), 10)
    plt_imshow("Draw Outline", output, figsize=(16, 10))


#사용자가 찍은 꼭짓점을 우상,좌상,좌하,우하순 정렬한다
def user_contours(points):
    # x 좌표와 y 좌표를 크기순으로 정렬하여 찾는다
    points.sort(key=lambda point: (point[0], point[1]))

    # x 좌표가 작은 두점 => 왼쪽 상하단
    top_left = min(points[:2], key=lambda point: point[1])
    bottom_left = max(points[:2], key=lambda point: point[1])

    # x 좌표가 큰 두점 => 오른쪽 상하단
    top_right = min(points[2:], key=lambda point: point[1])
    bottom_right = max(points[2:], key=lambda point: point[1])

    orderPoints = [top_right, top_left, bottom_left, bottom_right]

    # 차원 변환 
    user_contours = np.array([[[x, y]] for x, y in orderPoints], dtype=np.int32)
    return user_contours


