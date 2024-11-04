from transformers import pipeline

# GPT-2 기반 텍스트 생성 파이프라인 로드
generator = pipeline("text-generation", model="gpt2")

# 텍스트 생성
result = generator("Once upon a time", max_length=50, num_return_sequences=1)
print(result)