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



# vocab_size = 10000
# embed_dim = 128
# hidden_dim = 64
# output_dim = 7  
# seq_len = 20
# batch_size = 8

# # Model Oluşturuluyor
# model = TextRNN(vocab_size, embed_dim, hidden_dim, output_dim, rnn_type='LSTM', num_layers=2, bidirectional=True)

# input_tensor = torch.randint(0, vocab_size, (batch_size, seq_len))  # (8, 20)
# print(f"Input tensor shape: {input_tensor.shape}")

# # Forward Pass
# output = model(input_tensor)
# print(f" Output tensor shape: {output.shape}")