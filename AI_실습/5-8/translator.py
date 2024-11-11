import os
import requests
from dotenv import load_dotenv
from PyQt5 import QtWidgets
from PyQt5.QtCore import QUrl
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from pydub import AudioSegment
from pydub.playback import play
import io
import playsound

class TranslatorApp(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()

        # 번역 모델 로드
        model_name = "facebook/nllb-200-distilled-600M"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

        # API 설정
        load_dotenv()
        self.api_key = os.getenv("ElevenLabs_API_KEY")
        self.url = os.getenv("API_URL")

        # 음성 재생기
        self.player = QMediaPlayer()

    def init_ui(self):
        # UI 구성
        self.text_input = QtWidgets.QLineEdit(self)
        self.text_input.setPlaceholderText("번역할 텍스트 입력")
        self.translate_button = QtWidgets.QPushButton("번역 및 음성 생성", self)
        self.output_label = QtWidgets.QLabel(self)
        self.play_button = QtWidgets.QPushButton("음성 재생", self)
        self.play_button.setEnabled(False)

        # 레이아웃 설정
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.text_input)
        layout.addWidget(self.translate_button)
        layout.addWidget(self.output_label)
        layout.addWidget(self.play_button)
        self.setLayout(layout)

        # 버튼 클릭 시 이벤트 핸들러 연결
        self.translate_button.clicked.connect(self.translate_and_generate_audio)
        self.play_button.clicked.connect(self.play_audio)

        # 윈도우 창 설정
        self.setWindowTitle("번역 및 음성 생성기")
        self.show()

    def translate_and_generate_audio(self):
        text = self.text_input.text()

        # 번역 수행
        inputs = self.tokenizer(text, return_tensors="pt")
        generated_tokens = self.model.generate(inputs.input_ids, forced_bos_token_id=self.tokenizer.lang_code_to_id["kor_Hang"])
        translated_text = self.tokenizer.decode(generated_tokens[0], skip_special_tokens=True)

        # 음성 생성 요청
        data = {
            "text": translated_text,
            "model_id": "eleven_multilingual_v2",
            "voice_settings": {
                "stability": 0.5,
                "similarity_boost": 1,
                "style": 0.5,
                "use_speaker_boost": True
            }
        }
        headers = {
            "xi-api-key": self.api_key,
            "Content-Type": "application/json"
        }
        response = requests.post(self.url, json=data, headers=headers)

        if response.status_code == 200:
            output_audio_path = "C:/Users/82103/OneDrive/바탕 화면/ysooj.github.io/AI_실습/5-8/output_audio.mp3"
            with open(output_audio_path, "wb") as f:
                f.write(response.content)

            self.output_label.setText(f"번역 결과: {translated_text}")
            self.play_button.setEnabled(True)
        else:
            self.output_label.setText("음성 생성 실패")

    def play_audio(self):
        # 음성 파일 재생
        audio_path = "C:/Users/82103/OneDrive/바탕 화면/ysooj.github.io/AI_실습/5-8/output_audio.mp3"
        if os.path.exists(audio_path):
            # Pydub을 통해 mp3 파일을 불러와서 재생
            playsound.playsound(audio_path)
        else:
            self.output_label.setText("오디오 파일을 찾을 수 없습니다.")

if __name__ == '__main__':
    app = QtWidgets.QApplication([])
    translator = TranslatorApp()
    app.exec_()