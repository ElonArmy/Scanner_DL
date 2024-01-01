# Scanner_DL
- one-branch

## 해결 해야할 문제(우선 순위 순)
- Main Probs
  - 책 페이지 영역 탐지(boundary) => 세그먼테이션 
    - unet으로 왜곡된 페이지 영역 분리(U-Net: Convolutional Networks for Biomedical Image Segmentation)
    - https://joungheekim.github.io/2020/09/28/paper-review/
    - Transformer 모델로도 페이지 영역 분리가 가능한가(An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale)
    - classification에서는 우수하다 https://kmhana.tistory.com/27 
  - 페이지의 휘어짐(원근변환, 휘어진 3D의 원본을 이해하고 2D 이미지 보정)
    - 2D이미지를 3D로 예측 => 지도학습 (https://arxiv.org/pdf/1703.10131.pdf)
    - 위와 비슷해 보이는 방식(https://www3.cs.stonybrook.edu/~cvl/content/papers/2019/SagnikKe_ICCV19.pdf)
    - Pix2Pix 생성형 모델을 이용해 손실된 부분을 생성 또는 복원
    
- Sub Probs
  - 자동 스캔:
    버튼을 누르지 않고도 페이지를 자동스캔하여 저장하는 기능 (https://khurramjaved.com/RecursiveCNN.pdf)
  - 그림자 제거:
    촬영자의 손,기기 등의 그림자가 있을경우 그 영역을 인식해서 보정 하는 기능(https://www.kaggle.com/c/denoising-dirty-documents/data)
  - 손가락 제거:
    책 페이지를 고정하는 손가락을 인식하여 제거하는 기능 => 텍스트를 가릴 경우 OCR이 어려워진다 => "텍스트가 가려졌습니다" 알려주는 기능
  - 조명반사 제거:
    조명의 반사로 인해 책페이지가 하양게 붕 떠서 내용이 잘보이지 않을때 => 어떻게 해야할지 잘모르겠음(조사 필요)

- 역할 분담
  - 시뮬레이션을통한 데이터 확보
  - 전처리 및 데이터 증강
  - Dewarpnet 모델 구현
  - 모델학습 및 튜닝
  - 
- 목표 => 페이지의 휘어짐까지
- 어떤 모델을 사용해서 해결할지 탐구 및 논의
  
