## RCF 간단 설명
- RCF는 microsoft에서 만든 ResNet기반으로 edge detection을 위해 만들어진 모델이다
- ResNet은 기존 CNN 기반 이미지 인식모델에서 신경망이 깊어질수록 발생하는 그래이디언트 소실문제를 잔차 학습(residual learning)및 효율적인 파라미터 사용으로 완화하는 구조를 제시한것

## 가상환경 사용법(windows일 경우)
- macOS의 경우 방법이 다름
- 터미널에서 먼저 가상환경을 만든다 ex) python -m venv rcfenv
- 터미널에서 가상환경을 실행시킨다 ex) source rcfenv/Scripts/activate
- 가상환경에 패키지를 설치한다 ex) pip install -r requirements.txt
- RCF.ipynb 를 한 셀씩 실행시켜본다