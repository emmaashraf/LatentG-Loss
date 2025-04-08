import torch 
import torch.nn as nn

class TextRNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim, rnn_type='RNN', num_layers=3, bidirectional=True):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn_type = rnn_type.upper()
        self.bidirectional = bidirectional
        rnn_cls = {'RNN': nn.RNN, 'LSTM': nn.LSTM, 'GRU': nn.GRU}[self.rnn_type]

        self.rnn = rnn_cls(embed_dim, hidden_dim, num_layers=num_layers,
                           batch_first=True, bidirectional=bidirectional)

        direction_factor = 2 if bidirectional else 1
        self.fc = nn.Linear(hidden_dim * direction_factor, output_dim)

    def forward(self, x):
        x = self.embedding(x)
        if self.rnn_type == 'LSTM':
            _, (hn, _) = self.rnn(x)
        else:
            _, hn = self.rnn(x)

        if self.bidirectional:
            hn = torch.cat((hn[-2], hn[-1]), dim=1)
        else:
            hn = hn[-1]

        return self.fc(hn)
