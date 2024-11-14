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

# .env 파일에서 api 키 가져오기
API_KEY = os.getenv('sparta_api_key')

# API 키가 잘 로드되었는지 확인
if API_KEY is None:
    raise ValueError("API key is missing from .env file")

# 환경 변수에 API 키 설정
os.environ['OPENAI_API_KEY'] = API_KEY

# 모델 초기화
model = ChatOpenAI(model="gpt-4o-mini")

# PyPDFLoader 인스턴스를 생성하고 PDF 파일을 로드할 준비를 한다. 파일의 경로 입력
loader = PyPDFLoader("ch3/[2024 한권으로 ok 주식과 세금].pdf")

# PDF에서 텍스트 추출. 페이지 별 문서 로드
docs = loader.load()

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

# 벡터 임베딩 생성. OpenAI 임베딩 모델 초기화
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")