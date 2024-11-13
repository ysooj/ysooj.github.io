from sentence_transformers import SentenceTransformer
import numpy as np

# Multilingual-E5-large-instruct 모델 로드
model = SentenceTransformer('intfloat/multilingual-e5-large')

# 문장 리스트
sentences = [
    "참새는 짹짹하고 웁니다.",
    "LangChain과 Faiss를 활용한 예시입니다.",
    "자연어 처리를 위한 임베딩 모델 사용법을 배워봅시다.",
    "유사한 문장을 검색하는 방법을 살펴보겠습니다.",
    "강좌를 수강하시는 수강생 여러분 감사합니다!"
]

# 문장들을 임베딩으로 변환
embeddings = model.encode(sentences)

# 임베딩 벡터 출력
print(embeddings.shape)  # (4, 1024) - 4개의 문장이 1024 차원의 벡터로 변환됨
print(embeddings[0])