from transformers import pipeline

# 감정 분석 파이프라인 로드
sentiment_analysis = pipeline("sentiment-analysis")
result = sentiment_analysis("I love using Hugging Face!")
print(result)