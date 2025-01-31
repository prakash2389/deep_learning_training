
import torch
import torch.nn as nn
import torch.optim as optim

class ANN(nn.Module):
    def __init__(self):
        super(ANN, self).__init__()
        self.fl1 = nn.Linear(2, 3)
        self.fl2 = nn.Linear(3, 1)
    def forward(self, x):
        x = torch.relu(self.fl1(x))
        x = self.fl2(x)
        return x

model = ANN()

criterian = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

input = torch.tensor([[1.0,2.0],[3.0,4.0],[5.0,6.0]])
target =torch.tensor([[3.1],[7.1],[11.1]])

for epoch in range(100):
    # print(optimizer.param_groups)
    optimizer.zero_grad()
    output = model(input)
    loss = criterian(output, target)
    # print(loss)
    print(output)
    loss.backward()
    optimizer.step()
    if epoch % 2 == 0:
        # print(loss)
        pass



