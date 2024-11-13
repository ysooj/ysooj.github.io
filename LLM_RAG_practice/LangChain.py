import os
from getpass import getpass

os.environ["sparta_api_key"] = getpass("OpenAI API key 입력: ")

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

# 모델 초기화
model = ChatOpenAI(model="gpt-4")

# 모델에 메시지 전달
response = model.invoke([HumanMessage(content="안녕하세요, 무엇을 도와드릴까요?")])
print(response.content)
