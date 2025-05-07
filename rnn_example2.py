import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Dataset

# Sample dataset: Small snippet of Shakespeare's text
text = """
To be, or not to be, that is the question:
Whether 'tis nobler in the mind to suffer
The slings and arrows of outrageous fortune,
Or to take arms against a sea of troubles
And by opposing end them.
"""

# Character-level encoding
chars = sorted(set(text))
char_to_idx = {ch: i for i, ch in enumerate(chars)}
idx_to_char = {i: ch for i, ch in enumerate(chars)}

# Hyperparameters
seq_length = 40  # Input sequence length
batch_size = 64
hidden_size = 128
num_layers = 2
learning_rate = 0.005
num_epochs = 50


# Prepare dataset
class TextDataset(Dataset):
    def __init__(self, text, seq_length):
        self.text = text
        self.seq_length = seq_length
        self.data = [
            (text[i:i + seq_length], text[i + seq_length])
            for i in range(len(text) - seq_length)
        ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        seq, target = self.data[idx]
        x = torch.tensor([char_to_idx[ch] for ch in seq], dtype=torch.long)
        y = torch.tensor(char_to_idx[target], dtype=torch.long)
        return x, y


dataset = TextDataset(text, seq_length)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


# RNN Model
class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(RNNModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden):
        out, hidden = self.rnn(x, hidden)
        out = self.fc(out[:, -1, :])  # Take the output of the last time step
        return out, hidden


# Initialize model
input_size = len(chars)
output_size = len(chars)

model = RNNModel(input_size, hidden_size, num_layers, output_size)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    hidden = None  # Initialize hidden state
    for x_batch, y_batch in dataloader:
        # One-hot encode input
        x_onehot = torch.zeros(x_batch.size(0), x_batch.size(1), len(chars))
        for i, seq in enumerate(x_batch):
            x_onehot[i, torch.arange(seq.size(0)), seq] = 1.0

        # Forward pass
        outputs, hidden = model(x_onehot, hidden)
        loss = criterion(outputs, y_batch)

        # Backward pass and optimization
        optimizer.zero_grad()
        # loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# Save model
torch.save(model.state_dict(), "rnn_shakespeare.pth")
