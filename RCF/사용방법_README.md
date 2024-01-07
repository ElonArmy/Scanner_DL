## RCF 간단 설명
- RCF는 microsoft에서 만든 ResNet기반으로 edge detection을 위해 만들어진 모델이다
- ResNet은 기존 CNN 기반 이미지 인식모델에서 신경망이 깊어질수록 발생하는 그래이디언트 소실문제를 잔차 학습(residual learning)및 효율적인 파라미터 사용으로 완화하는 구조를 제시한것

## 가상환경 사용법(windows일 경우)
- macOS의 경우 방법이 다름
- pip을 최신버전으로 먼저 업데이트한다 => python.exe -m pip install --upgrade pip
- 터미널에서 먼저 가상환경을 만든다 ex) python -m venv rcfenv
- 터미널에서 가상환경을 실행시킨다 ex) source rcfenv/Scripts/activate
- 가상환경에 패키지를 설치한다 ex) pip install -r requirements.txt
- RCF.ipynb 를 한 셀씩 실행시켜본다

## 문제점
- 단순한 경계선 인식 모델로 사각형의 경계를 찾아 윤곽선(네모로) 보정을 한것임
- 때문에 왜곡된 문서의 경우 인식 불가함 ⇒ dewarp모델의 필요성
- CPU로 돌리면 느림 약 20초 이상 => 서버에서 그래픽으로 돌리면됨