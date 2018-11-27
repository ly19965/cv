
from src import detect_faces
from PIL import Image

image = Image.open('/data1/aipd_tuijian/charlesliu/dataset/V1/43444/Google_217.png')
bounding_boxes, landmarks = detect_faces(image)
print landmarks
print bounding_boxes

