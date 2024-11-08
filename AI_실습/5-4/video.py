import cv2  # OpenCV 라이브러리 임포트
from ultralytics import YOLO  # YOLO 모델 불러오기

# YOLOv8 모델을 로드합니다.
model = YOLO('yolov8n.pt')  # 또는 다른 YOLO 모델을 사용할 수 있습니다.

# 웹캠을 통해 실시간 객체 탐지
cap = cv2.VideoCapture(0)  # 웹캠 캡처 시작

# 웹캠을 통해 프레임 읽기 및 YOLO 모델로 객체 탐지 수행

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 프레임을 YOLOv8 모델에 입력하여 객체 탐지 수행
    results = model(frame)
		
    # 프레임을 YOLOv8 모델에 입력하여 객체 탐지 수행
    results = model(frame)

#     # 객체 탐지 결과를 화면에 표시
#     for box in results[0].boxes:
#         # 박스 좌표와 레이블을 가져옵니다.
#         x1, y1, x2, y2 = map(int, box.xyxy[0])  # 좌표를 정수로 변환
#         label = box.cls  # 클래스 레이블

#         # 객체 탐지 박스를 그립니다.
#         cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
#         cv2.putText(frame, str(label), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

#         # 프레임을 화면에 출력합니다.
#         cv2.imshow("YOLOv8 Detection", frame)
    
#     # 'q' 키를 누르면 종료
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # 웹캠 종료 및 창 닫기
# cap.release()
# cv2.destroyAllWindows()



# 웹캠 상에서도 어떤 객체를 탐지하고 있는 지 알고 싶어.
# 위의 코드로는 웹캠 상에서 각 바운딩 박스마다 tensor라고만 적혀있다.
# 아래의 코드로 실행하면 각 바운딩 박스마다 어떤 객체를 탐지하고 있는 지 나타나게 된다.
# 객체의 정보를 추출하여 화면에 표시
    for result in results:
        boxes = result.boxes  # 바운딩 박스 정보
        labels = result.names  # 탐지된 객체의 레이블 정보

        for box in boxes:
            # 각 객체의 바운딩 박스 좌표와 레이블 추출
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # 바운딩 박스 좌표
            label = labels[int(box.cls)]  # 객체 이름
            confidence = box.conf[0]  # 신뢰도

            # 객체의 바운딩 박스와 레이블을 프레임에 그리기
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # 바운딩 박스
            cv2.putText(frame, f"{label} {confidence:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)  # 레이블과 신뢰도

    # 프레임을 표시
    cv2.imshow("YOLOv8 Webcam", frame)

    # 'q' 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 웹캠 종료 및 창 닫기
cap.release()
cv2.destroyAllWindows()