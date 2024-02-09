# Scanner_DL
- one-branch

## DL Tasks (우선 순위 순)
- Main Tasks
  - Simple DewarpNet 모델 구현 
  - Dewarp 모델 핵심 구조(end-to-end 파이프라인)
    - 공간변형 네트워크(Spatial Transformer Network, STN) :  이미지에서 문서의 위치와 방향을 감지하고, 표준화된 형태로 변환해준다
    - 박막-스플라인(Thin-Plate-Spline,TPS) : 왜곡된 문서와 평평한 문서 사이에 일대일 대응이 있는 제어점에 대해 공간변형함수를 계산하여 기하하적 왜곡을 예측한다. 이때 제어점은 메쉬그리드로 정의한다 . 다시말해,  들어온 문서를 메쉬그리드로 예측하고 TPS는 이 예측된 메쉬그리드를 정규 메쉬그리드로 변환하여 문서를 편다.
    - Texture Mapping: 정규변환하여 펴진 3d 메쉬그리드에 이미지의 문서 텍스쳐를 매핑한다.
    - Refinement network: 그림자, 균일하지않은 조명들에의한 잡음제거 => 이진화를해도되고 푸리에 변환으로 고주파정보만 추출하여 사용해도됨
  - 모델 Serving API 서버 구축
    - API 정의
          
- Sub Tasks
    - 오토 스캔
        - 버튼을 누르지 않고도 페이지를 넘기면서 오토 스캔하여 저장하는 기능 (https://khurramjaved.com/RecursiveCNN.pdf)
    - OCR 과정: 문자인식

- 프로젝트 기간 늘어날지도 모름 각을 재봐야 함, 늘어나도 참여 가능한지 여부
- 목표 => 메인 Tasks 완료
  
