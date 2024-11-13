from transformers import pipeline

# GPT-2 기반 텍스트 생성 파이프라인 로드
generator = pipeline("text-generation", model="gpt2")

# 텍스트 생성
generated_text = generator("Once upon a time", max_length=50, num_return_sequences=1)

# 결과 출력
print(generated_text[0]['generated_text'])