# 파인튜닝을위해 따로 만들어둔 폴더이다
- 상위폴더의 requirements.txt로 환경을 맞추어주고
- [Enet](https://drive.google.com/file/d/1OVHETBHQ5u-1tnci3qd7OcAjas4v1xnl/view?usp=sharing), [Tnet](https://drive.google.com/file/d/1gEp4ecmdvKds2nzk9CaZb_pLvhRoyAsv/view?usp=sharing)
- 사전학습 모델을 models 폴더에 넣는다.
- 가상환경에 설치된 파이토치의 rcf_env/Lib/site-packages/torch/functional.py에서 line 504의 return을 수정해야됨
    ```
    return _VF.meshgrid(tensors, **kwargs, indexing="ij")
    ```
- 가상환경 실행
- 터미널을 열어 visdom실행
    ```
    python -m visdom.server -port 10086
    ```
- train.ipynb를 차례대로 실행시킨다 
- 파인튜닝된 모델은 chck에 저장된다.
- demo_test.ipynb에서 파인튜닝모델 실행 해볼수있다
- data폴더에서 diw 데이터를 추가하거나 bgtex 배경이미지를 추가하고 make_path.ipynb를 실행시켜 이름을 파싱하여 저장하고 원하는 데이터로 파인튜닝 해볼수있다. 
- x, xm, bg의 데이터 갯수는 동일해야한다.
- diw.py의 diwaug클래스에서의 타이트크롭을 조절하면 좀더 다양한 각도를 학습할수있을듯함

- 수정한부분:
    - doc3d데이터를 사용한 검증생략, doc3d데이터 손실함수 loss0 1로 고정
    - 데이터증강 생략
    - 배경을 불러오는 부분에 오류가있어서 bg 불러오는 코드 수정

- 개선해야할점:
    - doc3d데이터를 만들어 온전한 학습을 시도
    - 가독성 좋게 간단하게 코드 수정하기