from ultralytics import YOLO
import cv2
from PyQt5.QtWidgets import QApplication, QLabel, QVBoxLayout, QWidget, QPushButton
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QImage, QPixmap

class VideoCaptureWidget(QWidget):
    def __init__(self):
        super().__init__()

        # YOLOv8x 모델 로드 (YOLOv8x)
        self.model = YOLO('yolov8x.pt')

        # UI 설정
        self.setWindowTitle("실시간 객체 탐지")
        self.image_label = QLabel(self)
        self.layout = QVBoxLayout()
        self.layout.addWidget(self.image_label)

        self.start_button = QPushButton("Start Webcam", self)
        self.start_button.clicked.connect(self.start_webcam)
        self.layout.addWidget(self.start_button)

        self.stop_button = QPushButton("Stop Webcam", self)
        self.stop_button.clicked.connect(self.stop_webcam)
        self.layout.addWidget(self.stop_button)
        self.setLayout(self.layout)

        # 웹캠 초기화
        self.capture = None
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)

    def start_webcam(self):
        """웹캠을 시작하고, 타이머를 시작하여 프레임을 주기적으로 읽음"""
        self.capture = cv2.VideoCapture(0)  # 웹캠 장치 열기
        self.timer.start(20)  # 20ms마다 프레임 업데이트 (50fps)

    def stop_webcam(self):
        """웹캠을 중지하고 타이머를 멈춤"""
        self.timer.stop()
        if self.capture is not None:
            self.capture.release()

    def update_frame(self):
        """웹캠에서 프레임을 읽어와서 YOLO 객체 탐지를 수행한 후 UI에 표시"""
        ret, frame = self.capture.read()
        if ret:
            # YOLOv8 객체 탐지 수행
            results = self.model(frame)
            result = results[0]

            # 바운딩 박스가 포함된 이미지를 가져옴
            img_with_boxes = result.plot()

            # OpenCV 이미지를 QImage로 변환
            rgb_image = cv2.cvtColor(img_with_boxes, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)

            # QImage를 QLabel에 표시하기 위해 QPixmap으로 변환
            self.image_label.setPixmap(QPixmap.fromImage(convert_to_Qt_format))

    def closeEvent(self, event):
        """윈도우 닫을 때 웹캠 해제"""
        if self.capture is not None:
            self.capture.release()

if __name__ == "__main__":
    app = QApplication([])
    window = VideoCaptureWidget()
    window.show()
    app.exec_()