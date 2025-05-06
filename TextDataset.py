"""
import torch 
from torch.utils.data import Dataset, DataLoader
import numpy as np


class TextDataset(Dataset):
    def __init__(self, X, y):
        super().__init__()
        self.embeddings = torch.tensor(X, dtype=torch.float32)
        self.labels = torch.tensor(y, dtype=torch.long).unsqueeze(-1)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]
"""
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer
import pandas as pd

class TextDataset(Dataset):
    def __init__(self, filepath, tokenizer, max_length=256):
        # قراءة البيانات من الـ CSV
        self.data = pd.read_csv(filepath)
        
        self.texts = self.data['text'].tolist()  # افترض أن عمود النصوص اسمه 'text'
        self.labels = self.data['label'].tolist()  # افترض أن عمود التسميات اسمه 'label'
        
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        # التوكنيزر لتحويل النصوص إلى تمثيلات عددية
        encoded = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )

        return {
            'input_ids': encoded['input_ids'].squeeze(0),
            'attention_mask': encoded['attention_mask'].squeeze(0),
            'label': torch.tensor(label, dtype=torch.long)
        }

# تحميل التوكنيزر (مثال: BERT)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# مثال على كيفية استخدام الكلاس
dataset = TextDataset(filepath='./final_data.csv', tokenizer=tokenizer)
