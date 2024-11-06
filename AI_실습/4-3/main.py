from fastapi import FastAPI

# FastAPI 인스턴스 생성
app = FastAPI()

# 루트 경로에 GET 요청이 들어왔을 때 "Hello World!"를 반환하는 엔드포인트 정의
@app.get("/test")
def read_root():
    return {"message": "hello test"}

# 터미널에 명령어 입력할 때
# uvicorn your_script_name:app --reload
# 위의 명령어를 터미널에 실행하면, IP 주소나 포트, 상태 등 다양한 로그가 나온다.
# 그런데 그 전에, 해당 파일이 있는 디렉토리로 이동해줘야 한다.
# 해당 파일은 C:\Users\82103\OneDrive\바탕 화면\ysooj.github.io\AI_실습\4-3에 있으므로
# cd "C:\Users\82103\OneDrive\바탕 화면\ysooj.github.io\AI_실습\4-3"
# 위의 코드와 같이 디렉토리를 먼저 이동해준 후에 reload 명령어를 실행해야 한다.
# 그리고 해당 파일의 이름은 main.py이므로
# reload 명령어의 your_script_name 부분을 main으로 고쳐서 실행해야 한다.

@app.get("/test2")
def read_root():
    return {"message": "hello test2"}