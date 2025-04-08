import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from collections import Counter
from torch.nn.utils.rnn import pad_sequence
from RNNBasedModel import TextRNN


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


def evaluate_metrics(model, dataloader, device):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

    return acc, precision, recall, f1


def main(json_path, model_path="text_gru_model.pt"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    X_test_tok = [x.lower().split() for x in data['X_test']]
    label_encoder = LabelEncoder()
    label_encoder.fit(data['y_train']) 

    y_test = label_encoder.transform(data['y_test'])

    vocab = build_vocab([x.lower().split() for x in data['X_train']])  
    X_test_enc = encode(X_test_tok, vocab)

    test_dataset = TextDataset(X_test_enc, y_test)
    test_loader = DataLoader(test_dataset, batch_size=64, collate_fn=collate_fn)

    model = TextRNN(vocab_size=len(vocab), embed_dim=128, hidden_dim=64,
                    output_dim=len(label_encoder.classes_),rnn_type="LSTM", num_layers=3) # rnn_type : RNN, LSTM, GRU 
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)

    acc, precision, recall, f1 = evaluate_metrics(model, test_loader, device)

    print(f"Test Results:")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")


if __name__ == "__main__":
    main("ma_data.json")
