import torch

# Create a 1D tensor
x = torch.tensor([1.0, 2.0, 3.0])
x.shape

# Create a 2D tensor
y = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
y.shape

# Create a tensor filled with zeros
z = torch.zeros(3, 3, 5)
z.shape

# Create a tensor with random values
random_tensor = torch.rand(3, 2)
random_tensor

a = torch.tensor([1,2,3])
b= torch.tensor([3,4,5])
print(a+b)
print(a-b)
print(a*b)

# Element-wise addition
torch.tensor([[1,2,3],[3,4,5]]) * torch.tensor([[1,2,3],[3,4,5]]).size

# Matrix multiplication
torch.matmul(torch.tensor([[1,2,3],[3,4,5]]) , torch.tensor([[1,2,3],[3,4,5],[4,5,6]]))

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Move tensor to GPU
x = torch.tensor([1.0, 2.0, 3.0]).to(device)

# 4. Autograd and Backpropagation

# Enable gradient computation
x = torch.tensor([1., 2., 3.], requires_grad=True)
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)

# Perform operations
y = x * x
z = y * y * 3

# Compute gradients
z.backward(torch.tensor([1.0, 1.0, 1.0]))
x.grad

12x3

24, 192, 648

y.backward(torch.tensor([2, 2, 2]))
x.grad

4, 8, 12
#neuralnetwork

import torch
import torch.nn as nn
import torch.optim as optim

class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(2, 3)
        self.fc2 = nn.Linear(3, 3)
        self.fc3 = nn.Linear(3, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = SimpleNN()

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr =0.01)

# Dummy data
inputs = torch.tensor([[1.0, 2.0], [3.0, 4.0],[5.0, 6.0]])
targets = torch.tensor([[1.0], [2.0], [3.0]])


for epoch in range(100):
    # Zero the gradients
    optimizer.zero_grad()

    # Forward pass
    outputs = model(inputs)

    # Calculate the loss
    loss = criterion(outputs, targets)

    # Backward pass (compute gradients)
    loss.backward()

    # Update the weights
    optimizer.step()

    if epoch % 10 ==0:
        print(f'Epoch {epoch} and Loss: {loss.item()}')

