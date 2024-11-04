from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# NLLB-200 모델과 토크나이저 로드
model_name = "facebook/nllb-200-distilled-600M"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# 번역할 문장
sentence = "The quick brown fox jumps over the lazy dog"

# NLLB-200에서 영어(Latin)와 한국어(Hangul) 코드 설정
inputs = tokenizer(sentence, return_tensors="pt")

# 입력 문장에 대한 번역 수행 (영어 -> 한국어)
generated_tokens = model.generate(inputs.input_ids, forced_bos_token_id=tokenizer.convert_tokens_to_ids("kor_Hang"), max_length=30)


# 번역 결과를 디코딩
translated_text = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)

print(f"Translated text: {translated_text}")