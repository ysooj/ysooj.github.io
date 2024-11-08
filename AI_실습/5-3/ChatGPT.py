# 강의에서는 이렇게 적었다.
# from openai import OpenAI

# client = OpenAI()

# completion = client.chat.completions.create(
#   model="gpt-4o",
#   messages=[
#     {"role": "system", "content": "너는 변호사야 나에게 법적인 상담을 해줘"},
#     {"role": "user", "content": "안녕하세요 저는 배형호입니다."}  
#   ]
# )

# print("Assistant: " + completion.choices[0].message.content)

import openai
import os
import time

# 환경 변수에서 API 키 불러오기
openai.api_key = os.getenv("OPENAI_API_KEY")

try:
    # API 호출 (gpt-3.5-turbo 모델 사용)
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",  # gpt-4 대신 gpt-3.5-turbo 사용
        messages=[
            {"role": "system", "content": "너는 변호사야. 나에게 법적인 상담을 해줘."},
            {"role": "user", "content": "안녕하세요 저는 배형호입니다."}
        ]
    )
    # 결과 출력
    print("Assistant: " + response['choices'][0]['message']['content'])

except openai.error.RateLimitError:
    print("API 사용 한도를 초과했습니다. 잠시 후 다시 시도해주세요.")
    time.sleep(60)  # 1분 후 재시도

# 코드를 실행하면 API 사용 한도를 초과했습니다. 잠시 후 다시 시도해주세요. 라고 출력된다.
# API를 결제해야 한다. 코드는 정상적으로 동작한다고 한다.



