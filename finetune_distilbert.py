from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import evaluate
import numpy as np
from transformers import DataCollatorWithPadding
import json
from transformers import BertTokenizer
from torch.utils.data import Dataset, DataLoader
from transformers import DataCollatorWithPadding
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import torch
from transformers import EarlyStoppingCallback

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, label2id, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.label2id = label2id  
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx], 
            truncation=True, 
            padding='max_length', 
            max_length=self.max_length,
            return_tensors='pt'
        )
        item = {key: encoding[key].squeeze(0).to(device) for key in encoding}  
        label = self.label2id[self.labels[idx]]
        item['labels'] = torch.tensor(label, dtype=torch.long).to(device) 
        
        return item

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(axis=-1)
    
    accuracy = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

model_path = "distilbert-base-uncased"

id2label = {
    0: "Anxiety", 
    1: "Depression", 
    2: "Stress", 
    3: "Suicidal",
    4: "Normal",
    5: "Bipolar",
    6: "Personality disorder",
}
label2id = {v: k for k, v in id2label.items()}

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=7, id2label=id2label, label2id=label2id).to(device) 


for name, param in model.base_model.named_parameters():
    param.requires_grad = True


with open("ma_data.json", "r", encoding="utf-8") as f:
    data = json.load(f)

X_train, y_train = data['X_train'], data['y_train']
X_test, y_test = data['X_test'], data['y_test']

train_dataset = TextDataset(X_train, y_train, tokenizer, label2id)
test_dataset = TextDataset(X_test, y_test, tokenizer, label2id)


train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=8)


data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


training_args = TrainingArguments(
    output_dir='./results/distilbert/train1',
    num_train_epochs=5, 
    per_device_train_batch_size=8, 
    per_device_eval_batch_size=8,   
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    fp16=True,  
    evaluation_strategy="epoch",  
    learning_rate=3e-5,  
    load_best_model_at_end=True,  
    metric_for_best_model="accuracy", 
    save_strategy="epoch"
)

early_stopping_callback = EarlyStoppingCallback(early_stopping_patience=2)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    callbacks=[early_stopping_callback],  
)


trainer.train()

print("Evaluating...")
results = trainer.evaluate()
print("Evaluation Results:", results)
