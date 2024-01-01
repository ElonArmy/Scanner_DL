# Scanner_DL
- one-branch

## 해결 해야할 문제(우선 순위 순)
- Main Probs
  - 책 페이지 Segmetation: 입력 이미지에서  책 페이지(또는 서류, 이하 문서)를 배경과 분리하고 영역을 예측 하는 것
    - 예상 문제
      - 이미지의 화질이 좋지 않을 경우 => 입력 이미지의 화질 제한을 둔다. 너무 낮을 경우 팝업알림
      - 문서와 배경의 유사할 경우 => 조사 필요 또는 인식 할 수 없다는 팝업알림
      - 전처리 과정을 수립 하여야함 => 그레이 스케일 및 리사이징 방법
  - 문서의 굴곡 평활화: 입력 이미지에서 문서의 3-D 형태를 복원하고 이를 바탕으로 Dewarping 하는 것(지도학습)
    - 해당 기법이 소개된 논문 (https://arxiv.org/pdf/1703.10131.pdf)
    - 해당 기법과 유사한 방식으로 문서를 평활화한 논문 (https://www3.cs.stonybrook.edu/~cvl/content/papers/2019/SagnikKe_ICCV19.pdf)
    - 예상 문제:
      - DewarpNet 논문에서는 이미지 모델 구조안에서 전처리하여 sementation과 dewarping을 같이 하는 듯하다
      - train데이터와 label 데이터를 어떻게 정의하고 확보할 것인가 => 3-D 모델링 툴을 활용하여 시뮬레이션으로 데이터 확보
        - 참고: blender로 책의 리깅을 구현하는영상(https://www.youtube.com/watch?v=O3wLE4ZscRw)
        - 정형화된 데이터를 안정적으로 얻을 수 있는 툴이나 방법을 더 알아 보아야함
      - 모델 학습의 시간 => 저렴한 코랩으로 해보고 aws나 google cloud의 GPU 활용
      - 문서의 두께에 따른 굴곡 변화 => 시뮬레이션으로 데이터 생성시 고려하여야함
      - 문서의 규격이 달라지는 문제 => 보편적인 문서의 다양한 규격을 학습시킨다.
    
- Sub Probs
  - 그림자 제거:
    촬영자의 손,기기 등의 그림자가 있을경우 그 영역을 인식해서 보정 하는 기능(https://www.kaggle.com/c/denoising-dirty-documents/data)
  - 손가락 제거:
    책 페이지를 고정하는 손가락을 인식하여 제거하는 기능
    - 예상 문제
      - 텍스트를 가릴 경우 OCR이 어려워진다 => "텍스트가 가려졌습니다" 알려주는 기능
  - 조명반사 제거:
    조명의 반사로 인해 책페이지가 하양게 붕 떠서 내용이 잘보이지 않을때 => 어떻게 해야할지 잘모르겠음(조사 필요)
    - openCV로 Binary Thresholding 기법으로 문자의 가독성 향상 (https://www.youtube.com/watch?v=tYF3EBkvYO0)
  - 자동 스캔:
    버튼을 누르지 않고도 페이지를 자동스캔하여 저장하는 기능 (https://khurramjaved.com/RecursiveCNN.pdf)
  - OCR 과정: 문자인식
  - 문서내 이미지 인식
    
- Side Probs
  - 백엔드와 통신 어떻게 할 것 인지 => 멀티시트, 이미지 패스 등
  - 프로젝트 기간 늘어날지도 모름 각을 재봐야 함, 늘어나도 참여 가능한지 여부
  - 프론트(Android)에게 구현할 기능과 필요한 UI를 정의 해주어야함
    
- 역할 분담
  - 다양한 상황을 가정하여 3-D툴을 활용하여 시뮬레이션을 통한 학습용 데이터 확보
  - 학습 데이터 및 입력 데이터 정의 및 전처리 및 데이터 증강, 출력 데이터 정의
  - Dewarpnet 모델 구현과 튜닝 , 모델 학습 => Doc3D(https://github.com/cvlab-stonybrook/DocIIW) 데이터셋으로 성능 평가


- 목표 => 일단 메인 문제들을 해결 해보는 것
  
