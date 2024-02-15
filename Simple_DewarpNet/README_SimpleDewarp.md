# Simple Dewarp 모델에 관한 설명
## 목표
- PaperEdge 모델을 사전학습 시켜 사용해본다.
- 사전학습을 위한 마스크 데이터 생성 -> 그랩컷 활용
- 사전학습 모델 [Enet](https://drive.google.com/file/d/1OVHETBHQ5u-1tnci3qd7OcAjas4v1xnl/view?usp=sharing), [Tnet](https://drive.google.com/file/d/1gEp4ecmdvKds2nzk9CaZb_pLvhRoyAsv/view?usp=sharing)

- 완벽 분석이 끝나고 나면 모델 아키텍처를 어떻게 바꾸면 더 좋은 성능이 나올수 있을지 테스트하면서 분석해본다


## Dewarp 모델 핵심 구조(end-to-end 파이프라인)
- 공간변환 네트워크(Spatial Transformer Networks, STNs) : 이미지에서 문서의 위치와 방향을 감지하고, 표준화된 형태로 변환해준다
- 박막-스플라인(Thin-Plate-Spline,TPS) : 왜곡된 문서와 평평한 문서 사이에 일대일 대응이 있는 제어점에 대해 공간변환함수를 계산하여 기하하적 왜곡을 예측한다. 이때 제어점은 메쉬그리드로 정의한다 . 다시말해,  들어온 문서를 메쉬그리드로 예측하고 TPS는 이 예측된 메쉬그리드를 정규 메쉬그리드로 변환하여 문서를 편다.
- Texture Mapping: 정규변환하여 펴진 3d 메쉬그리드에 이미지의 문서 텍스쳐를 매핑한다.
- Refinement network: 그림자, 균일하지않은 조명들에의한 잡음 => 이진화를해도되고 푸리에 변환으로 고주파정보만 추출하여 사용해도됨

