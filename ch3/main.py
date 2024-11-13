import os   # 운영체제와 상호작용하게 해주는 표준 라이브러리
from dotenv import load_dotenv  # .env 파일을 쉽게 로드할 수 있게 해주는 라이브러리
from langchain_google_genai import ChatGoogleGenerativeAI   # 

# .env 파일 로드
load_dotenv()

# .env 파일에서 api 키 가져오기
api_key = os.getenv('gemini_api_key')   # os.getenv : 환경 변수에서 값을 가져오는 역할을 한다. 여기서는 api 키 값을 가져왔다.
print(api_key)

# 모델 로드하기
model = ChatGoogleGenerativeAI(model="gemini-pro")