import os
import torch
import torch.nn as nn 
import torchvision 
import torchvision.transforms as transforms 

class NerualNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NerualNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn = nn.BatchNorm1d(hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x 

def feedforward_nerual_network():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    input_size = 784
    hidden_size = 500
    num_classes = 10
    num_epochs = 5
    batch_size = 100
    learning_rate = 0.001

    model_dir = 'model'
    os.makedirs(model_dir, exist_ok=True)
    model_weight = model_dir + '/feedforward_nerual_network_model.pth'
    mnist_path = '/media/weixing/diskD/dataset/pytorch/mnist'
    train_dataset = torchvision.datasets.MNIST(root=mnist_path, train=True, transform=transforms.ToTensor())#, download=True)
    test_dataset = torchvision.datasets.MNIST(root=mnist_path, train=True, transform=transforms.ToTensor())
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    model = NerualNet(input_size, hidden_size, num_classes).to(device)
    #model.load_state_dict(torch.load(model_weight)) 
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    total_step = len(train_loader)
    for epoch in range(num_epochs):
        avg_loss = 0
        for i, (images, labels) in enumerate(train_loader):
            images = images.reshape(-1,28*28).to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            avg_loss += loss.item()
            if (i+1)%100 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                    .format(epoch+1, num_epochs, i+1, total_step, loss.item()))#,end="\r")
        #print('  ',end="\r",flush=True)
    #print('Epoch [{}/{}], Loss: {:.8f}' .format(epoch+1, num_epochs, avg_loss/total_step))

    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.reshape(-1, 28*28).to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        print('Accuracy of the network on the 10000 test images: {} %'.format(100 * correct / total))
    torch.save(model.state_dict(), model_weight)

def feedforward_nerual_network_predict():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_size = 784
    hidden_size = 500
    num_classes = 10
    batch_size = 100
    model_dir = 'model'
    os.makedirs(model_dir, exist_ok=True)
    model_weight = model_dir + '/feedforward_nerual_network_model.pth'
    mnist_path = '/media/weixing/diskD/dataset/pytorch/mnist'
    train_dataset = torchvision.datasets.MNIST(root=mnist_path, train=True, transform=transforms.ToTensor())
    test_dataset = torchvision.datasets.MNIST(root=mnist_path, train=True, transform=transforms.ToTensor())
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    model = NerualNet(input_size, hidden_size, num_classes).to(device)
    model.load_state_dict(torch.load(model_weight))
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.reshape(-1, 28*28).to(device)
            labels = labels.to(device)
            outputs = model(images)
            value, index = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (index == labels).sum().item()
        print('Accuracy of the network on the 10000 test images: {} %'.format(100 * correct / total))


if __name__ == "__main__":
    #feedforward_nerual_network()
    feedforward_nerual_network_predict()