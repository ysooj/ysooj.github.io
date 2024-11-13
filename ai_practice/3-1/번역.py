from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

# M2M100 모델과 토크나이저 로드
model_name = "facebook/m2m100_418M"
tokenizer = M2M100Tokenizer.from_pretrained(model_name)
model = M2M100ForConditionalGeneration.from_pretrained(model_name)

# 번역할 문장
sentence = "The quick brown fox jumps over the lazy dog"

# 입력 문장을 토큰화
encoded_sentence = tokenizer(sentence, return_tensors='pt')

# 번역 대상 언어를 한국어로 설정 (M@M100은 직접 언어 코드를 설정해야 함)
tokenizer.src_lang = "en"
model.config.forced_bos_token_id = tokenizer.get_lang_id("ko")

# 번역 수행 (영어 -> 한국어)
generated_tokens = model.generate(**encoded_sentence)

# 번역 결과를 디코딩
translated_text = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)

print(f"Translated text: {translated_text}")