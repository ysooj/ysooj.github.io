# 사용환경 준비
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
import faiss
from langchain_community.vectorstores import FAISS

load_dotenv()

API_KEY = os.getenv('sparta_api_key')

if API_KEY is None:
    raise ValueError("API key is missing from .env file")

os.environ['OPENAI_API_KEY'] = API_KEY

# 모델 로드하기
model = ChatOpenAI(model="gpt-4o-mini")

# 문서 로드하기
loader = PyPDFLoader("ch3/[2024 한권으로 ok 주식과 세금].pdf")

docs = loader.load()

# 문서 청크로 나누기 - RecursiveCharacterTextSplitter
recursive_text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100, # 최대 100자씩 나눈다.
    chunk_overlap=10,   # 각 조각에 앞뒤로 5자의 중복을 포함한다.
    length_function=len,    # 길이를 계산할 때 문자 수(len)를 기준으로 한다.
    is_separator_regex=False,   # 구분자를 단순 문자열로 처리
)

splits_RCT = recursive_text_splitter.split_documents(docs)

print(splits_RCT[:10])  # 청킹된 내용 상위 10개 출력

# 벡터 임베딩 생성. OpenAI 임베딩 모델 초기화
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

# 벡터 스토어 생성
vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)

# FAISS를 Retriever로 변환
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 1})

# 프롬프트 템플릿 정의
contextual_prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer the question using only the following context."),
    ("user", "Context: {context}\\n\\nQuestion: {question}")
])

# RAG 체인 구성
class DebugPassThrough(RunnablePassthrough):
    def invoke(self, *args, **kwargs):
        # 부모 클래스의 invoke 메서드를 호출하여 처리된 결과를 받아온다.
        output = super().invoke(*args, **kwargs)
        # 처리된 결과를 출력하여 디버깅 용도로 확인한다.
        print("Debug Output:", output)
        # 처리된 결과를 그대로 반환한다.
        return output

class ContextToText(RunnablePassthrough):
    def invoke(self, inputs, config=None, **kwargs):  # config 인수도 받을 수 있도록 설정
        # context의 각 문서를 텍스트로 결합한다.
        context_text = "\n".join([doc.page_content for doc in inputs["context"]])
        # 결합된 텍스트와 사용자 질문을 함께 반환한다.
        return {"context": context_text, "question": inputs["question"]}

rag_chain_debug = {
    "context": retriever,                  # retriever는 context를 가져오는 단계다.
    "question": DebugPassThrough()         # DebugPassThrough는 question을 그대로 전달하며 디버깅을 출력한다.
} | DebugPassThrough() | ContextToText() | contextual_prompt | model  # 각 단계에 디버깅과 텍스트 변환을 추가한 파이프라인

# 챗봇 구동 확인
while True:  
    print("========================")
    
    query = input("질문을 입력하세요 (종료하려면 'exit'를 입력하세요): ")

    if query.lower() == 'exit':
        print("프로그램을 종료합니다.")
        break

    response = rag_chain_debug.invoke(query)  
    
    print("Final Response:")
    
    print(response.content)