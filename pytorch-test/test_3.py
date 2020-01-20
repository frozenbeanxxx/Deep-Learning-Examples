import torch
import torch.optim as optim
import torch.nn as nn 
import torch.nn.functional as F 

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.conv2 = nn.Conv2d(6, 16, 3)
        self.fc1 = nn.Linear(16*6*6, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
    
    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2,2))
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s 
        return num_features

def base_info():
    net = Net()
    print(net)
    params = list(net.parameters())
    print(len(params))
    for i in range(len(params)):
        print(params[i].size())

def random_input():
    input = torch.randn(1,1,32,32)
    #input = torch.randn(1,1,64,64)
    #print(input)
    net = Net()
    out = net(input)
    #print(out)
    print(out.size())
    target = torch.randn(10)
    target = target.view(1,-1)
    criterion = nn.MSELoss()
    loss = criterion(out, target)
    print(loss)
    print(loss.grad_fn)  # MSELoss
    print(loss.grad_fn.next_functions[0][0])  # Linear
    print(loss.grad_fn.next_functions[0][0].next_functions[0][0])  # ReLU

    net.zero_grad()
    print('conv1.bias.grad before backward')
    print(net.conv1.bias.grad)
    loss.backward()
    print('conv1.bias.grad after backward')
    print(net.conv1.bias.grad)

def update_weights():
    net = Net()
    optimizer = optim.SGD(net.parameters(), lr=0.01)
    optimizer.zero_grad()
    input = torch.randn(1,1,32,32)
    output = net(input)
    target = torch.randn(10)
    target = target.view(1,-1)
    criterion = nn.MSELoss()
    loss = criterion(output, target)
    print(net.conv1.data.grad)
    loss.backward()
    print(net.conv1.data.grad)
    optimizer.step()
    print(net.conv1.data.grad)


if __name__ == "__main__":
    #base_info()
    #random_input()
    update_weights()