import torch
import torch.nn as nn
from torchtext.vocab import GloVe


"""
Encoder modeult for Seq2Seq model
contains
1) GloveEmbedding
2) LSTMEncoder (bidirectional)
3) BertEncoder
"""
SPL_TOKEN = ["<pad>", "<unk>", "<sos>", "<eos>"]

class GloveEmbeddings():
    def __init__(self, embed_dim,  word_to_idx):
        self.embed_dim = embed_dim
        self.word_to_idx = word_to_idx
        self.spl_tokens = SPL_TOKEN
        self.vocab_size = len(word_to_idx)

    def get_embedding_matrix(self):
        # Load pre-trained GloVe embeddings
        glove = GloVe(name='6B', dim=self.embed_dim)
        embedding_matrix = torch.zeros((self.vocab_size, self.embed_dim))

        embedding_matrix[0] = torch.zeros(self.embed_dim)    # Padding token
        for i in range(1,len(SPL_TOKEN)):            
            embedding_matrix[i] = torch.randn(self.embed_dim)    # Start-of-sentence token
            
        for k, v in self.word_to_idx.items():
            if k in SPL_TOKEN:
                continue
            else:            
                if k in glove.stoi:
                    embedding_matrix[v] = torch.tensor(glove.vectors[glove.stoi[k]])
                else:
                    embedding_matrix[v] = embedding_matrix[1]
        return embedding_matrix


class LSTMEncoder(nn.Module):
    # Bidirectional LSTM Encoder
    def __init__(self, input_size , embed_dim, hidden_units =1024, num_layers = 1, embedded_matrix = None, dropout_prob = 0.5):
        super(LSTMEncoder, self).__init__()
        # parameters
        self.input_size = input_size
        self.embed_dim = embed_dim
        self.hidden_units = hidden_units
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout_prob)
        self.embedded_matrix = embedded_matrix
        self.embedding = nn.Embedding(input_size, embed_dim)
        if embedded_matrix is not None:
            self.embedding = nn.Embedding.from_pretrained(embedded_matrix)
        else:
            self_embedding  = nn.Embedding(input_size, embed_dim, padding_idx = 0)

        self.LSTM = nn.LSTM(input_size=embed_dim, hidden_size=hidden_units, num_layers = num_layers, batch_first = True, bidirectional = True)
        self.hidden = nn.Linear(2*hidden_units, hidden_units)   
        self.cell = nn.Linear(2*hidden_units, hidden_units)
        

    def forward(self,x):
        # apply dropout to the embeddings 
        x = self.dropout(self.embedding(x))
        # apply LSTM
        output, (hidden,cell) = self.LSTM(x)
        hidden = self.hidden(torch.cat((hidden[0:1], hidden[1:2]), dim = 2)) # concatenate the forward and backward LSTM hidden states
        cell = self.cell(torch.cat((cell[0:1], cell[1:2]), dim = 2)) # concatenate the forward and backward LSTM cell states
        return output, (hidden, cell)