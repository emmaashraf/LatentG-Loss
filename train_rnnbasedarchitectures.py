import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from collections import Counter
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from RNNBasedModel import TextRNN
from sklearn.preprocessing import LabelEncoder


class TextDataset(Dataset):
    def __init__(self, X, y):
        self.X = X 
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.long), torch.tensor(self.y[idx], dtype=torch.long)

def collate_fn(batch):
    texts, labels = zip(*batch)
    lengths = torch.tensor([len(t) for t in texts])
    texts_padded = pad_sequence(texts, batch_first=True)
    return texts_padded, torch.stack(labels)


def build_vocab(tokenized_texts, min_freq=1):
    counter = Counter(token for tokens in tokenized_texts for token in tokens)
    vocab = {'<PAD>': 0, '<UNK>': 1}
    for token, freq in counter.items():
        if freq >= min_freq:
            vocab[token] = len(vocab)
    return vocab

def encode(tokenized_texts, vocab):
    return [[vocab.get(token, vocab['<UNK>']) for token in tokens] for tokens in tokenized_texts]


def train(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    all_preds, all_labels = [], []

    for X_batch, y_batch in tqdm(dataloader):
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y_batch.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    return total_loss / len(dataloader), acc

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            total_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    return total_loss / len(dataloader), acc


def main(json_path, num_epochs=100, batch_size=64):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    
    X_train_tok = [x.lower().split() for x in data['X_train']]
    X_test_tok = [x.lower().split() for x in data['X_test']]

    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(data['y_train'])
    y_test = label_encoder.transform(data['y_test'])

    num_classes = len(label_encoder.classes_)

    vocab = build_vocab(X_train_tok)
    X_train_enc = encode(X_train_tok, vocab)
    X_test_enc = encode(X_test_tok, vocab)


    train_dataset = TextDataset(X_train_enc, y_train)
    test_dataset = TextDataset(X_test_enc, y_test)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=collate_fn)

    model = TextRNN(vocab_size=len(vocab), embed_dim=128, hidden_dim=64, output_dim=num_classes, rnn_type= "GRU",  num_layers=3) # rnn_type  : RNN, LSTM, GRU
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        train_loss, train_acc = train(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, test_loader, criterion, device)

        print(f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")

    
    torch.save(model.state_dict(), "text_gru_model.pt")
    print("Model saved.")

main("ma_data.json")  