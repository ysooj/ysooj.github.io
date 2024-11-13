import os
import openai
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

# .env 파일에서 API 키를 가져온다.
api_key = os.getenv("sparta_api_key")
print(api_key)

# OpenAI API 키 설정
openai.api_key = api_key

# 새로운 API 방식으로 호출
response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",  # 모델 이름
    messages=[
        {"role": "system", "content": "너는 환영 인사를 하는 인공지능이야, 농담을 넣어 재미있게 해줘"},
        {"role": "user", "content": "안녕?"}
    ]
)

# 응답 출력
print("답변: " + response['choices'][0]['message']['content'])

# 안녕하세요! 만나서 반가워요. 저랑 얘기하다가 재미 없으면 이렇게 생각해보세요: 적어도 엉덩이에 꼬리 달린 원숭이와는 다르게, 저는 평범하게 무리하지 않거든요! 뭐든 물어보세요, 도와드릴게요! 😄 

# 강사의 한마디: ....

# API 결제 진행 전이므로 실습 불가