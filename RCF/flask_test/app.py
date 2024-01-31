from flask import Flask, jsonify, request, send_file, after_this_request
from inference import get_prediction, get_points_and_perspective_transform
import cv2
import numpy as np
import os, io
import tempfile

app = Flask(__name__)

def get_latest_file(image_directory):
    # 디렉토리 내의 모든 파일 경로 가져온다
    all_files = [os.path.join(image_directory, f) for f in os.listdir(image_directory)]

    # 파일만 고른다
    all_files = [f for f in all_files if os.path.isfile(f)]

    # 수정시간에따라 정렬
    all_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)

    # 가장최근거 선택
    return all_files[0] if all_files else None

def delete_file(file_path):
    # 파일이 존재하는지 확인하고 삭제
    if os.path.exists(file_path):
        os.remove(file_path)
        print(f"파일 '{file_path}' 삭제됨.")
    else:
        print(f"파일 '{file_path}'을(를) 찾을 수 없습니다.")

# 포스트 요청만 허용한다
# @app.route('/predict', methods=['POST'])
# def predict():
#     # return 'Hello World!'
#     if request.method == 'POST':
#         # 이미지 파일을 받아온다
#         file = request.files['file']
#         # 바이트 형식으로 읽는다
#         img_bytes = file.read()
#         predicted_points = get_prediction(img_bytes)
        
#         # [[2200, 0], [2135, 0], [2157, 32], [2158, 80]] 형태
#         return jsonify(predicted_points) 
    

# 이미지를 받아서 좌표를 반환하는 함수
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' in request.files:
        file = request.files['file']
        
        # 저장할 경로 설정
        image_directory = './path_to_save_images'  
        if not os.path.exists(image_directory):
            os.makedirs(image_directory)

        image_path = os.path.join(image_directory, file.filename)

        # 이미지 파일 저장
        file.save(image_path)
        print('파일저장했음')
        
        # 저장된 이미지를 읽어서 처리
        img = cv2.imread(image_path)

        predicted_points = get_prediction(img)
        
        return jsonify(predicted_points)
    else:
        return jsonify({'error': 'No file found'}), 400
    
    
# 수정된 좌표를 받아와서 원근변환된 문서를 반환해주는 함수 
@app.route('/perspective_transform', methods=['POST'])
def perspective_transform():
    if request.method == 'POST':
        # 수정된 좌표받기 
        points = request.json.get('points')
        
        image_directory = './path_to_save_images'
        latest_file = get_latest_file(image_directory)
        
        img = cv2.imread(latest_file)
        
        # 추론에서 이미지 원근변환하기 
        transformed_doc_array = get_points_and_perspective_transform(img, points)
        
        # 메모리 내에서 이미지 인코딩
        _, encoded_image = cv2.imencode('.jpg', transformed_doc_array)
        encoded_image_bytes = encoded_image.tobytes()
        
        # 인코딩된 이미지 데이터를 BytesIO 객체에 쓰기
        file_stream = io.BytesIO(encoded_image_bytes)
        file_stream.seek(0)  # 스트림 위치를 시작으로 리셋
        
        # 변환된 이미지를 바이너리 데이터로 전송
        return send_file(
            file_stream,
            mimetype='image/jpeg',
            as_attachment=True,
            download_name='transformed_image.jpg',
        )
    else:
        return jsonify({'status': 'error', 'message': 'Invalid request method'})
   


if __name__ == '__main__':
    app.run(debug=True)
    