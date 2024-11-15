import torch
import torch.nn as nn

class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(BiLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Define the Bidirectional LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        
        # Define the output layer
        self.fc = nn.Linear(hidden_size*2, output_size)  # Multiply by 2 because of bidirectionality
        
    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(x.device)  # Multiply by 2 because of bidirectionality
        c0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(x.device)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # Output shape: (batch_size, seq_length, hidden_size*2)
        
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out

# Example usage:
input_size = 10  # Dimensionality of input features
hidden_size = 20  # Number of features in the hidden state
num_layers = 2  # Number of LSTM layers
output_size = 1  # Dimensionality of the output

# Create an instance of the BiLSTM model
model = BiLSTM(input_size, hidden_size, num_layers, output_size)

# Generate some random input data for testing
input_data = torch.randn(80, 3, input_size)  # (batch_size, seq_length, input_size)

# Forward pass
output = model(input_data)
print("Output shape:", output.shape)
