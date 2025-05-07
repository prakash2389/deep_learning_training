import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.datasets import PennTreebank
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import DataLoader
import random


traindata = PennTreebank(split='train')
tokenizer = get_tokenizer('basic_english')

def data_process(traindata):
    data = [tokenizer(item) for item in traindata]
    return data

vocab = build_vocab_from_iterator(data_process(traindata), specials=['<unk>'])


text = """
To be, or not to be, that is the question:
Whether 'tis nobler in the mind to suffer
The slings and arrows of outrageous fortune,
Or to take arms against a sea of troubles
And by opposing end them.
"""

# Character-level encoding
chars = sorted(set(text))
char_to_idx = {char: i for i, char in enumerate(chars)}
idx_to_char = {i: char for i, char in enumerate(chars)}

# hyperparameters
seq_length = 4
batch_size = 10
hidden_size = 20
num_layers = 2
learning_rate = 0.01
num_epochs = 100


# Prepare dataset
def encode_text(text, char_to_idx):
    return [char_to_idx[char] for char in text]

def decode_text(encoded_text, idx_to_char):
    return [idx_to_char[idx] for idx in encoded_text]

def create_dataset(text, seq_length, char_to_idx):
    encoded_text = encode_text(text, char_to_idx)
    dataset = []
    for i in range(len(encoded_text) - seq_length):
        dataset.append(encoded_text[i:i+seq_length])
    return dataset

dataset = create_dataset(text, seq_length, char_to_idx)
random.shuffle(dataset)

train_size = int(0.8 * len(dataset))
train_data = dataset[:train_size]
test_data = dataset[train_size:]

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

# Define the model

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.rnn = nn.RNN(hidden_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden):
        out = self.embedding(x)
        out, hidden = self.rnn(out, hidden)
        out = self.fc(out)
        return out, hidden

    def init_hidden(self, batch_size):
        return torch.zeros(self.num_layers, batch_size, self.hidden_size)

model = RNN(len(chars), hidden_size, num_layers, len(chars))

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Train the model

for epoch in range(num_epochs):
    for i, data in enumerate(train_loader):
        inputs = torch.tensor(data)
        targets = torch.tensor(data)
        hidden = model.init_hidden(batch_size)
        optimizer.zero_grad()
        outputs, hidden = model(inputs, hidden)
        loss = criterion(outputs.view(-1, len(chars)), targets.view(-1))
        loss.backward()
        optimizer.step()
        if i % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item()}')

# Test the model
with torch.no_grad():
    for i, data in enumerate(test_loader):
        inputs = torch.tensor(data)
        targets = torch.tensor(data)
        hidden = model.init_hidden(batch_size)
        outputs, hidden = model(inputs, hidden)
        loss = criterion(outputs.view(-1, len(chars)), targets.view(-1))
        if i % 100 == 0:
            print(f'Test Step [{i+1}/{len(test_loader)}], Loss: {loss.item()}')
