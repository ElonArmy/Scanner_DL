img_PATH = '../img_data/normal.jpg'


'''전처리 => 모듈로 만들어야함''' 
import io
import torchvision.transforms as transforms
from PIL import Image
    

def transform_image(image_bytes):
    my_transforms = transforms.Compose([transforms.Resize(255),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize(
                                            [0.485, 0.456, 0.406],
                                            [0.229, 0.224, 0.225])])
    image = Image.open(io.BytesIO(image_bytes))
    return my_transforms(image).unsqueeze(0)

with open(img_PATH, 'rb') as f:
    image_bytes = f.read()
    tensor = transform_image(image_bytes=image_bytes)
    print(tensor)



'''모델 예측 => 모듈로 만들어야함'''  
import json
from torchvision import models
from RCF_model import RCF, device

model = RCF(device=device)

def get_prediction(image_bytes):
    tensor = transform_image(image_bytes=image_bytes)
    outputs = model.forward(tensor)
    _, y_hat = outputs.max(1)
    predicted_idx = str(y_hat.item())
    return imagenet_class_index[predicted_idx]

with open(img_PATH, 'rb') as f:
    image_bytes = f.read()
    print(get_prediction(image_bytes=image_bytes))
    
    
