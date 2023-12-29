# Scanner_DL
- one-branch

## 해결 해야할 문제(우선 순위 순)
- Probs
  - 책 페이지 영역 탐지(boundary) => 세그먼테이션
  - 페이지의 휘어짐(원근변환, 휘어진 3D의 원본을 이해하고 2D 이미지 보정) => https://arxiv.org/pdf/1703.10131.pdf (2D이미지를 3D로 예측 => 지도학습)
  (https://www3.cs.stonybrook.edu/~cvl/content/papers/2019/SagnikKe_ICCV19.pdf)
  - 자동 스캔() https://khurramjaved.com/RecursiveCNN.pdf
  - 그림자 제거 (https://www.kaggle.com/c/denoising-dirty-documents/data)
  - 손가락 제거
  - 조명반사 제거
  
- 목표 => 페이지의 휘어짐까지
- 어떤 모델을 사용해서 해결할지 탐구 및 논의
  