from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset
import torch

# IMDb 데이터셋 로드
dataset = load_dataset("imdb")

# 훈련 및 테스트 데이터셋 분리
train_dataset = dataset['train'].shuffle(seed=42).select(range(1000))  # 1000개 샘플로 축소
test_dataset = dataset['test'].shuffle(seed=42).select(range(500))  # 500개 샘플로 축소

# BERT 토크나이저 로드
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# 데이터셋 토크나이징 함수 정의
def tokenize_function(examples):
    return tokenizer(examples['text'], padding="max_length", truncation=True)

# 데이터셋 토크나이징 적용
train_dataset = train_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)

# 모델 입력으로 사용하기 위해 데이터셋 포맷 설정
train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

# BERT 모델 로드
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# 훈련 인자 설정
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    evaluation_strategy="epoch",
    save_steps=10_000,
    save_total_limit=2,
)

# 트레이너 설정
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

# 모델 훈련
trainer.train()
trainer.evaluate()

import numpy as np
from sklearn.metrics import accuracy_score

# 평가 지표 함수 정의
def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)  # 예측된 클래스
    labels = p.label_ids  # 실제 레이블
    acc = accuracy_score(labels, preds)  # 정확도 계산
    return {'accuracy': acc}

# 이미 훈련된 트레이너에 compute_metrics를 추가하여 평가
trainer.compute_metrics = compute_metrics

# 모델 평가 및 정확도 확인
eval_result = trainer.evaluate()
print(f"Accuracy: {eval_result['eval_accuracy']:.4f}")