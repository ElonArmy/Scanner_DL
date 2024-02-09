# 모델 소스코드 분석하는 곳
- 모델 구조와 프로세스를 분석할것임
## 목록
- DocUnet  : Unet기반으로한 첫 디워핑 모델 
- RectiNet : DocUnet에서 gated network와 분기된 unet구조를 추가하여 개선
- PaperEdge: Enet(=globalwarp)으로 경계를 잡고 전반적인 문서골곡을 디포메이션한다.  Tnet(=localwarp)으로 나머지 텍스트의 굴곡등을 텍스트 정렬을 분석하여 정밀하게 포인트를 제어하여 디워핑

## 사용법
- Docunet:
- RectiNet: [코랩 데모](https://colab.research.google.com/drive/1aBFOIAZ5JHaoQsw4ihC0usZP0ZI-jlLE?usp=sharing)
- PaperEdge:
    - [Enet](https://drive.google.com/file/d/1OVHETBHQ5u-1tnci3qd7OcAjas4v1xnl/view?usp=sharing), [Tnet](https://drive.google.com/file/d/1gEp4ecmdvKds2nzk9CaZb_pLvhRoyAsv/view?usp=sharing)
    - 사전학습 모델을 models 폴더에 넣는다.
    - 가상환경을 키고 model_analysis/PaperEdge 경로에서 demo.py실행
    - --image_path를 수정해서 원하는 이미지 넣기
    ```
    python demo.py --img_path 'images/IMG_2453.jpg' --out_dir 'output'
    ```
    - output 폴더에서 결과물 확인

    - 가상환경에 설치된 파이토치의 rcf_env/Lib/site-packages/torch/functional.py에서 line 504의 return을 수정해야됨
    ```
    return _VF.meshgrid(tensors, **kwargs, indexing="ij")
    ```
