from flask import Flask, jsonify, request
from inference import get_prediction, transform_image

app = Flask(__name__)

# @app.route('/')
# def hello():
#     return 'Hello world'

# 포스트 요청만 허용한다
@app.route('/predict', methods=['POST'])
def predict():
    # return 'Hello World!'
    if request.method == 'POST':
        # 이미지 파일을 받아온다
        file = request.files['file']
        # 바이트 형식으로 읽는다
        img_bytes = file.read()
        class_id, class_name = get_prediction(image_bytes=img_bytes)
        return jsonify({'class_id': class_id, 'class_name': class_name})
    