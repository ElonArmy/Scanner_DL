# Scanner_DL
- one-branch

## 해결 해야할 문제(우선 순위 순)
- Probs
  - 책 페이지 영역 탐지(boundary) => 세그먼테이션
    - unet으로 왜곡된 페이지 영역 분리(U-Net: Convolutional Networks for Biomedical Image Segmentation)
    - https://joungheekim.github.io/2020/09/28/paper-review/
    - Transformer 모델로도 페이지 영역 분리가 가능한가(An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale)
    - classification에서는 우수하다 https://kmhana.tistory.com/27 
  - 페이지의 휘어짐(원근변환, 휘어진 3D의 원본을 이해하고 2D 이미지 보정)
    - 2D이미지를 3D로 예측 => 지도학습 (https://arxiv.org/pdf/1703.10131.pdf)
    - 위와 비슷해 보이는 방식(https://www3.cs.stonybrook.edu/~cvl/content/papers/2019/SagnikKe_ICCV19.pdf)
    - Pix2Pix 생성형 모델을 이용해 손실된 부분을 생성 또는 복원
  - 자동 스캔() https://khurramjaved.com/RecursiveCNN.pdf
  - 그림자 제거 (https://www.kaggle.com/c/denoising-dirty-documents/data)
  - 손가락 제거
  - 조명반사 제거
  
- 목표 => 페이지의 휘어짐까지
- 어떤 모델을 사용해서 해결할지 탐구 및 논의
  