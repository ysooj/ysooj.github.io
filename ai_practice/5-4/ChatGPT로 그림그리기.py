import openai
import os
from dotenv import load_dotenv

# .env 파일 로드
# .env 파일에서 API 키를 가져온다.
load_dotenv()
api_key = os.getenv("sparta_api_key")
print(api_key)

# OpenAI API 키 설정
openai.api_key = api_key

prompt = input("Prompt: ")

response = openai.Image.create(
    model = "dall-e-3",
    prompt = prompt,
    size = "1024x1024",
    quality = "hd",
    n = 1
)

image_url = response.data[0].url
print(image_url)

# 그러나 오류 메시지 Billing hard limit, 즉 결제 한도 문제로 결과가 출력되지 않는다.
# 코드는 문제없이 동작한다고 한다.