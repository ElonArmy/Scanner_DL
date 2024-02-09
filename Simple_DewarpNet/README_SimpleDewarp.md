# Simple Dewarp 모델에 관한 설명
- DewarpNet(2019)와 FDRnet(2022)의 구조를 참고하여 핵심 네트워크만을 가지고 간단하게 구현하는것이 목표이다
- PaperEdge 모델의 아키텍처를 참고하여 구현한다.
- 데이터 생성은 PaperEdge모델과 docUnet모델을 참고한다
- 현재는 따로 만들고있는데 최종적으로 하나의 모델안에 합쳐야한다.
- 사전학습 모델 [Enet](https://drive.google.com/file/d/1OVHETBHQ5u-1tnci3qd7OcAjas4v1xnl/view?usp=sharing), [Tnet](https://drive.google.com/file/d/1gEp4ecmdvKds2nzk9CaZb_pLvhRoyAsv/view?usp=sharing)

## Dewarp 모델 핵심 구조(end-to-end 파이프라인)
- 공간변환 네트워크(Spatial Transformer Networks, STNs) : 이미지에서 문서의 위치와 방향을 감지하고, 표준화된 형태로 변환해준다
- 박막-스플라인(Thin-Plate-Spline,TPS) : 왜곡된 문서와 평평한 문서 사이에 일대일 대응이 있는 제어점에 대해 공간변환함수를 계산하여 기하하적 왜곡을 예측한다. 이때 제어점은 메쉬그리드로 정의한다 . 다시말해,  들어온 문서를 메쉬그리드로 예측하고 TPS는 이 예측된 메쉬그리드를 정규 메쉬그리드로 변환하여 문서를 편다.
- Texture Mapping: 정규변환하여 펴진 3d 메쉬그리드에 이미지의 문서 텍스쳐를 매핑한다.
- Refinement network: 그림자, 균일하지않은 조명들에의한 잡음 => 이진화를해도되고 푸리에 변환으로 고주파정보만 추출하여 사용해도됨

