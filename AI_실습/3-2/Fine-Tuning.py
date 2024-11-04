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
train_dataset = train_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)

# 모델 입력으로 사용하기 위해 데이터셋 포맷 설정
train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
