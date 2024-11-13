from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset
import torch
import numpy as np
from sklearn.metrics import accuracy_score

dataset = load_dataset("imdb")

test_dataset = dataset["test"].shuffle(seed=42).select(range(500))

# BERT 토크나이저 로드
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# 데이터셋 토크나이징 함수 정의
def tokenize_function(examples):
    return tokenizer(examples['text'], padding="max_length", truncation=True)

# 데이터셋 토크나이징 적용
test_dataset = test_dataset.map(tokenize_function, batched=True)

# 모델 입력으로 사용하기 위해 데이터셋 포맷 설정
test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

# BERT 모델 로드
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# 모델을 평가모드로 바꾼다.
model.eval()

# 예측 및 평가
all_preds = []
all_labels = []

for batch in torch.utils.data.DataLoader(test_dataset, batch_size=8):
    with torch.no_grad():
        outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
    logits = outputs.logits
    preds = np.argmax(logits.numpy(), axis=1)
    all_preds.extend(preds)
    all_labels.extend(batch['label'].numpy())

# 정확도 계산
accuracy = accuracy_score(all_labels, all_preds)
print(f"Accuracy without fine-tuning: {accuracy:.4f}")
