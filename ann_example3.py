import torch, torchvision
from torch import nn
from torch import optim

# Create a 1D tensor
x = torch.tensor([1.0, 2.0, 3.0])
print(x)
print(x.shape)

if torch.cuda.is_available():
    x = x.to("cuda")

# ReLU: output = max(0, input)
# Sigmoid: output = 1 / (1 + exp(-input))
# Tanh: output = (exp(input) - exp(-input)) / (exp(input) + exp(-input))
# Softmax: output = exp(input) / sum(exp(input))
# LogSoftmax: output = log(softmax(input))
# CrossEntropyLoss: loss = -sum(target * log(softmax(input)))
# MSELoss: loss = sum((target - input)^2)
# BCELoss: loss = -sum(target * log(input) + (1 - target) * log(1 - input))



class ANN(nn.Module):
    def __init__(self):
        super(ANN, self).__init__()
        self.fc1 = nn.Linear(2,5)
        self.fc2 = nn.Linear(5,5)
        self.fc3 = nn.Linear(5,1)

    def forward(self,x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


model = ANN()

loss = nn.MSELoss()
optimzer = optim.Adam(model.parameters(), lr=0.01)

inputs = torch.tensor([[1,2], [2,3],[3,4], [4,5], [5,6]], dtype=torch.float32)
targets = torch.tensor([3,5,7,9,11], dtype=torch.float32)

for i in range(500):
    optimzer.zero_grad()
    outputs = model(inputs)
    loss_value = loss(outputs, targets)
    loss_value.backward()
    optimzer.step()

model(torch.tensor([8,9], dtype=torch.float32))

from torchvision import datasets, transforms
# Transform to normalize data
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True)
import matplotlib.pyplot as plt

# Function to show images
def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# Get a batch of training images
dataiter = iter(trainloader)
images, labels = next(dataiter)

import numpy as np
# Show a few images
imshow(torchvision.utils.make_grid(images[:100]))

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)  # 1 input channel, 6 output channels, 5x5 kernel
        self.pool = nn.MaxPool2d(2, 2)  # 2x2 pooling
        self.conv2 = nn.Conv2d(6, 16, 5)  # 6 input channels, 16 output channels, 5x5 kernel
        self.fc1 = nn.Linear(16 * 4 * 4, 120)  # 16 channels, 4x4 image, 120 output neurons
        self.fc2 = nn.Linear(120, 84)  # 120 input neurons, 84 output neurons
        self.fc3 = nn.Linear(84, 10)  # 84

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = torch.relu(self.fc1(x.view(-1, 16 * 4 * 4)))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = CNN()

loss = nn.CrossEntropyLoss()
optimzer = optim.Adam(model.parameters(), lr=0.1)

for i in range(10):
    model.train()

    for images, labels in trainloader:
        optimzer.zero_grad()
        outputs = model(images)
        loss_value = loss(outputs, labels)
        loss_value.backward()
        optimzer.step()
    print(loss_value)

# Test the model
testset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=1000, shuffle=False)

model.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in testloader:
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy: %d %%' % (100 * correct / total))



import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import MNIST, CIFAR10
from torch.utils.data import DataLoader

# Transform to normalize data
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

trainset = CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=32, shuffle=True)
testset= CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = DataLoader(testset, batch_size=1000, shuffle=False)

datatrain = iter(trainloader)
images, labels = next(datatrain)

# Function to show images
def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

import numpy as np
# Show a few images
imshow(torchvision.utils.make_grid(images[:100]))

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)  # 3 input channel, 6 output channels, 5x5 kernel
        self.pool = nn.MaxPool2d(2, 2)  # 2x2 pooling
        self.conv2 = nn.Conv2d(6, 16, 5)  # 6 input channels, 16 output channels, 5x5 kernel
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # 16 channels, 5x5 image, 120 output neurons
        self.fc2 = nn.Linear(120, 84)  # 120 input neurons, 84 output neurons
        self.fc3 = nn.Linear(84, 10)  # 84

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = torch.relu(self.fc1(x.view(-1, 16 * 5 * 5)))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class RNN():
    def __init__(self):
        self.rnn = nn.RNN(10, 20, 2)
        self.fc = nn.Linear(20, 10)

    def forward(self, x):
        x, _ = self.rnn(x)
        x = self.fc(x)
        return x

rnn = RNN()
x = torch.randn(100, 32, 10)
output = rnn(x)
print(output.shape)


