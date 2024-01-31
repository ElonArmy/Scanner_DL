'''모델예측:
1. 이미지가 입력되면 경계선을 예측하고 꼭짓점을 찾아 반환해주는 모듈이다.
2. 사용자가 확인후 수정된(또는 수정 되지않으니) 좌표를 다시 받아 영역을 크롭후 
   원근변환한 이미지를 다시 서버에 보내준다.  
'''

# img_PATH = '../img_data/normal.jpg'

import cv2
import numpy as np
import json
from PIL import Image
from RCF_model import RCF, device, find_contours,four_point_transform
# from RCF_model import plt_imshow


model = RCF(device=device)

# 이미지를 불러온다
# img = cv2.imread(img_PATH)

def get_prediction(img):
    # 모델에 넣어 경계선 이미지 받는다.
    edge_img = model.detect_edge(img)

    # 경계선이미지의 꼭짓점을 찾는다
    # 모양 (np.array[[[2200,    0]], [[2135,    0]], [[2157,   32]], [[2158,   80]]])
    contours = find_contours(edge_img, 1)
    
    # 모양 바꾸기
    contours = [item[0] for item in contours]

    predicted_points = \
    [[int(item) for item in np.array(contour, dtype=np.int32)] for contour in contours]
    
    # [[2200, 0], [2135, 0], [2157, 32], [2158, 80]] 형태로 서버로 보낼것이다
    return predicted_points
 
# 사용자가 predicted_points 좌표를 확인하고(수정이 될수도있다) 다시 받으면 이미지 원근변환 실행
# [[714, 461], [2505, 644],[223, 3553], [2488, 3654]] => np.array형식 3차원으로 다시 변환하여
# 이런식으로 들어온다. 정렬되지않음
def get_points_and_perspective_transform(img, modified_points):
   # numpy 3차원으로 변환
   modified_points = np.array(modified_points, dtype=int).reshape(-1, 1, 2)
   transformed_doc_array = four_point_transform(img, modified_points.reshape(4, 2))
   # plt_imshow("Receipt Transform", dewarped_doc)  # 이미지확인
   # dewarped_doc_img = Image.fromarray(transformed_doc_array, 'RGB')
   # 이미지 저장
   # dewarped_doc_img.save('./dewarped/dewarped_doc_img.jpg')
   return transformed_doc_array
    
    
