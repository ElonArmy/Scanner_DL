# Scanner_DL
- one-branch

## DL Tasks (우선 순위 순)
- Main Tasks
  - Dewarp 모델 핵심 구조(end-to-end 파이프라인) 구현
    - 공간변형 네트워크(Spatial Transformer Network, STN) :  이미지에서 문서의 위치와 방향을 감지하고, 표준화된 형태로 변환해준다
    - 박막-스플라인(Thin-Plate-Spline,TPS) : 왜곡된 문서와 평평한 문서 사이에 일대일 대응이 있는 제어점에 대해 공간변형함수를 계산하여 기하하적 왜곡을 예측한다. 이때 제어점은 메쉬그리드로 정의한다 . 다시말해,  들어온 문서를 메쉬그리드로 예측하고 TPS는 이 예측된 메쉬그리드를 정규 메쉬그리드로 변환하여 문서를 편다.
    - Texture Mapping: 정규변환하여 펴진 3d 메쉬그리드에 이미지의 문서 텍스쳐를 매핑한다.
    - Refinement network: 그림자, 균일하지않은 조명들에의한 잡음제거 => 이진화를해도되고 푸리에 변환으로 고주파정보만 추출하여 사용해도됨
  - 모델 Serving API 서버 구축
    - API 정의
    - 백엔드와 통신 어떻게 할 것 인지 => 멀티시트, 이미지 패스 등
          
- Sub Tasks
    - 그림자 제거(모델내에서 처리함)
        - openCV로 Binary Thresholding 기법으로 문자의 가독성 향상 (https://www.youtube.com/watch?v=tYF3EBkvYO0)
        - 푸리에 고속변환하여 고주파 정보만 사용하는 방법
    - 손가락 제거
        - 책 페이지를 고정하는 손가락을 인식하여 제거하는 기능
        - 예상 문제
            - 텍스트를 가릴 경우 OCR이 어려워진다 => "텍스트가 가려졌습니다" 알려주는 기능
    - 오토 스캔
        - 버튼을 누르지 않고도 페이지를 넘기면서 오토 스캔하여 저장하는 기능 (https://khurramjaved.com/RecursiveCNN.pdf)
    - OCR 과정: 문자인식
    - 문서내 이미지 인식
- Side Tasks
    - flutter로 모바일 스캐너앱 간단구현
    - 프로젝트 기간 늘어날지도 모름 각을 재봐야 함, 늘어나도 참여 가능한지 여부
- 목표 => 메인 Tasks 완료
  
