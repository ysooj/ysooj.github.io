import os
import requests
from pydub import AudioSegment
from pydub.playback import play
import io
from dotenv import load_dotenv
import playsound

# .env 파일 로드
load_dotenv()

# .env 파일에서 API 키를 가져온다.
api_key = os.getenv("ElevenLabs_API_KEY")
url = "https://api.elevenlabs.io/v1/text-to-speech/9BWtsMINqrJLrRacOk9x"
# https://elevenlabs.io/docs/api-reference/get-voice
# 여기서 ElevenLabs의 url을 받아올 수 있다. Get Voic의 GET에 있는 것을 복사하면 된다.
# 그러나 해당 URL은 https://api.elevenlabs.io/v1/voices/{voice_id} 로 복사되는데,
# 이를 url에 저장하면 오류가 난다. 위의 코드처럼 text-to-speech를 voices로 바꾸고
# {voice_id} 대신 원하는 목소리의 voice_id를 넣어주면 된다.

# https://api.elevenlabs.io/v1/voices
# 여기는 voice_id를 받아올 수 있는 링크.

print(api_key)

# API 키와 URL이 없으면 종료
if not api_key:
    print("API_KEY가 설정되지 않았습니다. .env 파일을 확인하세요.")
    exit()

# 설정 가능한 변수
output_filename = "output_audio.mp3"

headers = {
    "xi-api-key": api_key,
    "Content-Type": "application/json"
}

# 문장을 입력받습니다.
text = input("텍스트를 입력하세요: ")

# 음성 생성 요청을 보냅니다.
# 각각의 최대값은 1이다.
data = {
    "text": text,
    "model_id": "eleven_multilingual_v2",
    "voice_settings": {
        "stability": 1,
        "similarity_boost": 1,
        "style": 1,
        "use_speaker_boost": True
    }
}

response = requests.post(url, json=data, headers=headers, stream=True)

if response.status_code == 200:
    audio_content = b""
    for chunk in response.iter_content(chunk_size=1024):
        if chunk:
            audio_content += chunk

    segment = AudioSegment.from_mp3(io.BytesIO(audio_content))
    segment.export(output_filename, format="mp3")
    print(f"Success! Wrote audio to {output_filename}")

    # 오디오를 재생합니다.
    playsound.playsound(output_filename)
else:
    print(f"Failed to save file: {response.status_code}")