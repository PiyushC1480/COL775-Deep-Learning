import torch
import torch.nn as nn

"""
Decoder modult for Seq2Seq model
contains
1) LSTMDecoder
"""

class LSTMDecoder(nn.Module):
    def __init__(self, input_size, embedding_dim, hidden_units = 1024, num_layers = 1, dropout_prob = 0.5):
        super(LSTMDecoder, self).__init__()
        # parameters
        self.input_size = input_size
        self.embedding_dim = embedding_dim
        self.hidden_units = hidden_units
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout_prob)
        self.embedding = nn.Embedding(input_size, embedding_dim)
        self.LSTM = nn.LSTM(input_size = embedding_dim, hidden_size = hidden_units, num_layers = num_layers, batch_first = True)
        self.fc = nn.Linear(in_features = hidden_units, out_features = input_size)
    def forward(self, x, hidden_cell):
        # apply dropout to the embeddings
        x = self.dropout(self.embedding(x))
        x = x.unsqueeze(1) # unsqueeze the embeddings to add a dimension
        # apply LSTM
        out, (h_t, c_t) = self.LSTM(x, hidden_cell)
        out = self.fc(out)
        return out, (h_t, c_t)
    

    