import os   # 운영체제와 상호작용하게 해주는 표준 라이브러리
from dotenv import load_dotenv  # .env 파일에서 환경 변수를 불러올 수 있게 해주는 라이브러리

from langchain_openai import ChatOpenAI # LangChain 라이브러리에서 OpenAI의 챗봇 모델을 사용하기 위한 코드.
# 즉, OpenAI의 챗 모델을 사용하여 언어 모델을 초기화하기 위한 코드
from langchain_core.messages import HumanMessage    # LangChain 라이브러리에서 HumanMessage 클래스를 임포트하는 코드
# 이 클래스는 LangChain의 메시지 처리 시스템에서 사용자 메시지를 나타내는 데 사용된다.
# LangChain은 대화형 AI 시스템을 구축할 때 여러 종류의 메시지를 처리하는데,
# HumanMessage는 사람이 보낸 메시지를 모델에 전달할 때 사용된다.
# HumanMessage 클래스는 LangChain에서 인간의 메시지를 모델로 전달하는 데 필요한 형태로 포맷팅해준다.

from langchain_community.document_loaders import PyPDFLoader  # LangChain 라이브러리에서 PDF 파일을 로드하는 데 사용하는 코드
# PyPDFLoader는 PDF 파일을 로드하여 텍스트를 추출하는 클래스다.
# PyPDFLoader를 langchain.document_loaders에서 임포트하는 것이 더 이상 권장되지 않기 때문에
# langchain_community.document_loaders에서 임포트해야 한다.

from langchain.text_splitter import CharacterTextSplitter
# langchain.text_splitter에서 CharacterTextSplitter 클래스를 가져온다.
# 이 클래스는 텍스트를 여러 덩어리로 분할하는 기능을 제공한다.
from langchain.text_splitter import RecursiveCharacterTextSplitter
# LangChain 라이브러리의 text_splitter 모듈에서 RecursiveCharacterTextSplitter 클래스를 가져온다.
# 긴 텍스트를 의미가 끊기지 않도록 일정한 길이의 조각으로 나누는 역할을 한다.

from langchain_openai import OpenAIEmbeddings   # langchain_openai 라이브러리에서 OpenAIEmbeddings 클래스를 불러온다.

import faiss    # 벡터 검색을 위한 라이브러리
from langchain_community.vectorstores import FAISS  # # langchain_community에서 FAISS 벡터스토어 클래스를 임포트한다.

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
# LangChain의 ChatPromptTemplate와 RunnablePassthrough 클래스를 임포트한다.
# 즉, langchain_core에서 필요한 모듈을 불러온다.

# .env 파일 로드
load_dotenv()   # .env 파일을 로드하여 환경 변수들을 설정한다.

# .env 파일에서 api 키 가져오기
API_KEY = os.getenv('sparta_api_key')   # os.getenv : 환경 변수에서 값을 가져오는 역할을 한다. 여기서는 api 키 값을 가져왔다.
# # .env 파일에 저장된 'sparta_api_key'라는 환경 변수의 값을 가져왔다.

# API 키가 잘 로드되었는지 확인
if API_KEY is None:
    raise ValueError("API key is missing from .env file")
# print(API_KEY)

# 환경 변수에 API 키 설정
os.environ['OPENAI_API_KEY'] = API_KEY
# 이렇게 설정하면 openai 라이브러리가 환경 변수를 사용한다.
# openai.api_key = API_KEY ==> 이건 옛날 버전이니 위의 코드를 사용하자.
# 이제 OpenAI 라이브러리가 자동으로 환경 변수를 사용하여 API 키를 설정한다.
# ChatOpenAI 모델을 초기화 할 때 API 키를 별도로 전달할 필요가 없다.

# import os
# from getpass import getpass
# os.environ["OPENAI_API_KEY"] = getpass("OpenAI API key 입력: ")
# 이 코드를 통해 api key가 잘 작동하는 지 확인할 수 있다.

# 모델 초기화
model = ChatOpenAI(model="gpt-4o-mini") # ChatOpenAI 객체를 초기화하여 사용할 모델을 설정한다.

# PyPDFLoader 인스턴스를 생성하고 PDF 파일을 로드할 준비를 한다. 파일의 경로 입력
loader = PyPDFLoader("ch3/[2024 한권으로 ok 주식과 세금].pdf")

# PDF에서 텍스트 추출. 페이지 별 문서 로드
docs = loader.load()    # loader.load()를 호출하여 PDF 파일의 텍스트를 페이지별로 불러온다.

# 아래 코드를 통해 추출된 텍스트를 확인해볼 수 있다.
# for doc in docs:
#     print(doc)

# 문서 청크로 나누기
# 1. CharacterTextSplitter
text_splitter = CharacterTextSplitter(
    separator="\n\n",   # 두 개의 개행 문자를 구분자로 사용
    chunk_size=100, # 최대 100자씩 나눈다.
    chunk_overlap=10,   # 각 조각에 앞뒤로 5자의 중복을 포함한다.
    length_function=len,    # 길이를 계산할 때 문자 수(len)를 기준으로 한다.
    is_separator_regex=False,   # 구분자를 단순 문자열로 처리
)

splits_CT = text_splitter.split_documents(docs)

print(splits_CT[:10]) # 청킹된 내용 상위 10개 출력

# 2. RecursiveCharacterTextSplitter
recursive_text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100, # 최대 100자씩 나눈다.
    chunk_overlap=10,   # 각 조각에 앞뒤로 5자의 중복을 포함한다.
    length_function=len,    # 길이를 계산할 때 문자 수(len)를 기준으로 한다.
    is_separator_regex=False,   # 구분자를 단순 문자열로 처리
)

splits_RCT = recursive_text_splitter.split_documents(docs)

print(splits_RCT[:10])  # 청킹된 내용 상위 10개 출력

# 벡터 임베딩 생성
# OpenAI 임베딩 모델 초기화
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
# 'text-embedding-ada-002' 모델을 사용하여 텍스트 데이터를 임베딩 벡터로 변환한다.
# 이 모델은 텍스트 데이터를 고차원 벡터로 변환하는 데 사용된다.
# 자연어 처리(NLP) 작업에서 텍스트를 벡터 공간으로 변환하여, 유사도 검색, 클러스터링, 분류 등 다양한 작업에 활용된다.

# 벡터 스토어 생성
vectorstore = FAISS.from_documents(documents=splits_RCT, embedding=embeddings)

# FAISS를 Retriever로 변환
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 1})

# 프롬프트 템플릿 정의
contextual_prompt = ChatPromptTemplate.from_messages([
    # 시스템 메시지: 모델에게 주어진 문맥 내에서만 질문에 답하라는 지시를 추가
    ("system", "Answer the question using only the following context."),
    
    # 사용자 메시지: 실제 문맥({context})과 질문({question})이 채워질 수 있도록 변수를 포함한 프롬프트 템플릿 생성
    # {context}와 {question}은 실제 사용자가 제공한 값으로 대체된다.
    ("user", "Context: {context}\\n\\nQuestion: {question}")
])

# RAG 체인 구성
# 프롬프트 템플릿은 위에서 정의했다.

# 'RunnablePassthrough' 클래스를 상속받은 DebugPassThrough 클래스 정의
# 이 클래스는 입력된 데이터를 그대로 전달하면서, 중간 결과를 디버깅 용도로 출력한다.
class DebugPassThrough(RunnablePassthrough):
    def invoke(self, *args, **kwargs):
        # 부모 클래스의 invoke 메서드를 호출하여 처리된 결과를 받아온다.
        output = super().invoke(*args, **kwargs)
        # 처리된 결과를 출력하여 디버깅 용도로 확인한다.
        print("Debug Output:", output)
        # 처리된 결과를 그대로 반환한다.
        return output

# 문서 리스트를 텍스트로 변환하는 ContextToText 클래스 정의
# 'RunnablePassthrough'를 상속받아, context의 각 문서를 텍스트로 결합하는 기능을 수행한다.
class ContextToText(RunnablePassthrough):
    def invoke(self, inputs, config=None, **kwargs):  # config 인수도 받을 수 있도록 설정
        # context의 각 문서를 텍스트로 결합한다.
        context_text = "\n".join([doc.page_content for doc in inputs["context"]])
        # 결합된 텍스트와 사용자 질문을 함께 반환한다.
        return {"context": context_text, "question": inputs["question"]}

# RAG 체인에서 각 단계에 DebugPassThrough 추가
# retriever는 'context'를 가져오는 역할을 하고, DebugPassThrough는 사용자 질문이 잘 전달되는지 확인한다.
# 이후 ContextToText가 문서 리스트를 텍스트로 변환하고, 마지막으로 프롬프트 템플릿을 사용하여 모델을 호출한다.
rag_chain_debug = {
    "context": retriever,                  # retriever는 context를 가져오는 단계다.
    "question": DebugPassThrough()         # DebugPassThrough는 question을 그대로 전달하며 디버깅을 출력한다.
} | DebugPassThrough() | ContextToText() | contextual_prompt | model  # 각 단계에 디버깅과 텍스트 변환을 추가한 파이프라인

# 챗봇 구동 확인
# 무한 루프를 시작. 이 루프는 사용자가 질문을 입력할 때마다 계속 반복된다.
while True:  
    # 사용자에게 질문을 입력하라는 메시지를 출력
    print("========================")
    
    # 사용자로부터 질문을 입력받음
    query = input("질문을 입력하세요: ")
    
    # 'rag_chain_debug' 체인을 호출하여 질문을 처리하고 응답을 받음
    # 이 때, 'query'는 사용자가 입력한 질문이다.
    response = rag_chain_debug.invoke(query)  
    
    # 'Final Response:'라는 메시지를 출력하여 최종 응답을 나타냄
    print("Final Response:")
    
    # 'response.content'는 모델이 반환한 응답의 내용을 출력한다.
    print(response.content)


# FAISS 말고 Chroma
# 이번 개인 과제에서 일반 GPT 결과랑 FAISS 결과랑 비교할 수 있을 것이다!