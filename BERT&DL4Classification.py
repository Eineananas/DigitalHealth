import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

import time
from datetime import datetime

file_path="/home/wtang/train_1006.xlsx"
df = pd.read_excel(file_path)

occupation_mapping = {
    0: "Non Skill",
    1: "Skill"
}

df['G'] = df['Label'].map(occupation_mapping)
df.dropna(subset=['sentence'], inplace=True)  
df.reset_index(drop=True, inplace=True)  
X_texts = df['sentence']
y = df['G']
y_encoded = pd.factorize(y)[0] 

X_train, X_val, y_train, y_val = train_test_split(X_texts, y_encoded, test_size=0.2, random_state=42)
print("The shape of X_train:", X_train.shape)


tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
model = BertForSequenceClassification.from_pretrained('bert-base-chinese', num_labels=len(occupation_mapping))
#用 bert-base-chinese 作为中文预训练基底
#BertForSequenceClassification 自动在 BERT 顶部加上一个分类层，num_labels=2，对应这个二分类任务
#如果没有接任何层的话，transformer会输出两个东西，A:每个token融合了上下文信息的词向量; b:每个句子的向量
class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


train_dataset = TextDataset(X_train.tolist(), y_train.tolist(), tokenizer, max_length=128)
val_dataset = TextDataset(X_val.tolist(), y_val.tolist(), tokenizer, max_length=128)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)


model.train()
# Model Training
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

for epoch in range(10):
    for batch in train_loader:
        optimizer.zero_grad()
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch + 1}, Loss: {loss.item()}')
    now = datetime.now()
    formatted_time = now.strftime("%Y-%m-%d %H:%M:%S")
    print("Local Time: ", formatted_time)


model.eval()
# Model Evaluation
preds, true_labels = [], []

with torch.no_grad():
    for batch in val_loader:
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        outputs = model(input_ids, attention_mask=attention_mask)
        preds.extend(torch.argmax(outputs.logits, dim=1).tolist())
        true_labels.extend(labels.tolist())


print(classification_report(true_labels, preds, target_names=list(occupation_mapping.values())))

model_path = '/home/wtang/transformer_model_CLASSIFICATION.pth'
torch.save(model.state_dict(), model_path)
print(f"Model saved to {model_path}")

torch.save(model, '/home/wtang/transformer_model_full_rewrite_oct3.pth')
print(f"Full Model saved!")
