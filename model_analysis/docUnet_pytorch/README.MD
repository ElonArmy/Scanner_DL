# DocUNet: Document Image Unwarping via A Stacked U-Net

[paper](https://www3.cs.stonybrook.edu/~cvl/content/papers/2018/Ma_CVPR18.pdf)

## 분석
- 목표는 왜곡된 이미지의 픽셀 $S(u, v)$ 을 펼쳐진 원복 이미지로 옮기는 매핑 필드를 만드는 것이다.
$$D(x, y) = S(u, v)$$
- 모델 학습을 위해 왜곡된 이미지와 이를 펼쳐주는 디포메이션 필드 데이터쌍이 필요하다.


