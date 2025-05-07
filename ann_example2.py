import torch
from torch import nn
from torch import optim

class ANN(nn.Module):
    def __init__(self):
        super(ANN, self).__init__()
        self.fc1 = nn.Linear(3,5)
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

inputs = torch.tensor([[1,2,3], [2,4,6],[3,6,9], [4,8,12], [5,10,15]], dtype=torch.float32)
targets = torch.tensor([1,2,3,4,5], dtype=torch.float32)

for i in range(100):
    optimzer.zero_grad()
    outputs = model(inputs)
    loss_value = loss(outputs, targets)
    loss_value.backward()
    optimzer.step()
    print(loss_value)
    print(model.fc1.weight.grad)

model(torch.tensor([100,200,300], dtype=torch.float32))

