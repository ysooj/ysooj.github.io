from ultralytics import YOLO
import cv2
from matplotlib import pyplot as plt

# YOLOv8 모델 로드 (YOLOv8s)
model = YOLO('yolov8n.pt')

# 이미지 파일 경로
image_path = image_path = "C:/Users/82103/OneDrive/바탕 화면/ysooj.github.io/AI_실습/5-4/cat.jpeg"

# 이미지를 모델에 입력하여 객체 탐지 수행
results = model(image_path)

# 탐지 결과 출력
result = results[0]

# 탐지된 객체들이 표시된 이미지를 가져옴
img_with_boxes = result.plot()  # result.plot()은 바운딩 박스가 포함된 이미지를 반환

# Matplotlib을 사용하여 이미지 출력
plt.imshow(cv2.cvtColor(img_with_boxes, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()