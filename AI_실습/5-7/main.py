from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import openai
from dotenv import load_dotenv
import os

# .env 파일에서 환경 변수 로드
load_dotenv()

app = FastAPI()

# 환경 변수에서 OpenAI API 키 가져오기
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("API 키가 설정되지 않았습니다. 환경 변수를 확인하세요.")

# OpenAI API 키 설정
openai.api_key = api_key

# Jinja2 템플릿 설정
templates = Jinja2Templates(directory="app/templates")

# 정적 파일 서빙
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# 초기 시스템 메시지 설정
system_message = {
    "role": "system",
    "content": "너는 환영 인사를 하는 인공지능이야, 농담을 넣어 재미있게해줘"
}

# 대화 내역을 저장할 리스트 초기화
messages = [system_message]

@app.get("/", response_class=HTMLResponse)
async def get_chat_page(request: Request):
    """채팅 페이지 렌더링"""
    conversation_history = [msg for msg in messages if msg["role"] != "system"]
    return templates.TemplateResponse("index.html", {"request": request, "conversation_history": conversation_history})

@app.post("/chat", response_class=HTMLResponse)
async def chat(request: Request, user_input: str = Form(...)):
    """사용자 메시지를 받아 OpenAI API 호출 및 응답 반환"""
    global messages

    # 사용자의 메시지를 대화 내역에 추가
    messages.append({"role": "user", "content": user_input})

    # OpenAI API 호출
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages
    )

    # AI의 응답 가져오기
    assistant_reply = completion.choices[0].message.content

    # AI의 응답을 대화 내역에 추가
    messages.append({"role": "assistant", "content": assistant_reply})

    # 화면에 표시할 대화 내역에서 system 메시지를 제외하고 전달
    conversation_history = [msg for msg in messages if msg["role"] != "system"]

    # 결과를 HTML로 반환 (대화 내역과 함께)
    return templates.TemplateResponse("index.html", {
        "request": request,
        "conversation_history": conversation_history
    })

# 이를 실행하려면 이 파일이 있는 디렉토리로 이동해서 uvicorn <파일명>:app --reload 명령어를 실행해야 한다.
# cd AI_실습
# cd 5-7
# cd app
# uvicorn main:app --reload
# 여기서 <파일명>에는 .py 확장자를 제외한 파일 이름을 넣어야 하기 때문에 그냥 main만 적어주었다.

# 나는 실행하면 창은 뜨지만, 메시지 칸에 메시지를 입력하고 전송을 누르면 오류가 난다.
# OpenAI API를 결제하지 않았기 때문이다.
# 코드는 정상적으로 동작한다고 한다.


