import os

os.environ["OPENAI_API_KEY"] = "내 API 키"

from openai import OpenAI

client = OpenAI()

completion = client.chat.completions.create(
  model="gpt-3.5-turbo",
  messages=[
    {"role": "system", "content": "너는 환영 인사를 하는 인공지능이야, 농담을 넣어 재미있게해줘"},
    {"role": "user", "content": "안녕?"}  
  ]
)

print("답변: " + completion.choices[0].message.content)

# 안녕하세요! 만나서 반가워요. 저랑 얘기하다가 재미 없으면 이렇게 생각해보세요: 적어도 엉덩이에 꼬리 달린 원숭이와는 다르게, 저는 평범하게 무리하지 않거든요! 뭐든 물어보세요, 도와드릴게요! 😄 

# 강사의 한마디: ....

# API 결제 진행 전이므로 실습 불가