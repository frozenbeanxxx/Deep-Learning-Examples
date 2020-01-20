import torch 
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

def bidirectional_recurrent_nerual_network():
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Hyper-parameters
    sequence_length = 28
    input_size = 28
    hidden_size = 128
    num_layers = 2
    num_classes = 10
    batch_size = 100
    num_epochs = 2
    learning_rate = 0.003

    mnist_path = r'E:\dataset\mnist'
    # MNIST dataset
    train_dataset = torchvision.datasets.MNIST(root=mnist_path,
                                            train=True, 
                                            transform=transforms.ToTensor(),
                                            download=True)

    test_dataset = torchvision.datasets.MNIST(root=mnist_path,
                                            train=False, 
                                            transform=transforms.ToTensor())

    # Data loader
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                            batch_size=batch_size, 
                                            shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                            batch_size=batch_size, 
                                            shuffle=False)
    class BiRNN(nn.Module):
        def __init__(self, input_size, hidden_size, num_layers, num_classes):
            super(BiRNN, self).__init__()
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
            self.fc = nn.Linear(hidden_size*2, num_classes)

        def forward(self, x):
            h0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(device)
            c0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(device)
            out, _ = self.lstm(x, (h0, c0))
            out = self.fc(out[:,-1,:])
            return out 

    model = BiRNN(input_size, hidden_size, num_layers, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    total_step = len(train_loader)
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            images = images.reshape(-1, sequence_length, input_size).to(device)
            labels = labels.to(device)
        
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if (i+1) % 100 == 0:
                print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                    .format(epoch+1, num_epochs, i+1, total_step, loss.item()))

    # Test the model
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.reshape(-1, sequence_length, input_size).to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print('Test Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total)) 

    # Save the model checkpoint
    model_path = 'model/weight_bidirectional_recurrent_nerual_network.ckpt'
    torch.save(model.state_dict(), model_path)

if __name__ == "__main__":
    bidirectional_recurrent_nerual_network()
