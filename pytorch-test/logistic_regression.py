import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

def lgistic_regression():
    input_size = 784
    num_classes = 10
    num_epochs = 5
    batch_size = 100
    learning_rate = 0.001

    model_weight = 'model/model_3.ckpt'
    mnist_path = 'E:/dataset/mnist'
    train_dataset = torchvision.datasets.MNIST(root=mnist_path, train=True, transform=transforms.ToTensor())#, download=True)
    test_dataset = torchvision.datasets.MNIST(root=mnist_path, train=True, transform=transforms.ToTensor())
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    # it = iter(test_loader)
    # a,b = next(it)
    # print(a,b)
    # return
    model = nn.Linear(input_size, num_classes)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    total_step = len(train_loader)
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            images = images.reshape(-1, 28*28)
            outputs = model(images)
            #print(outputs.detach().numpy())
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i+1) % 100 == 0:
                print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                        .format(epoch+1, num_epochs, i+1, total_step, loss.item()))

    with torch.no_grad():
        correct = 0
        total = 0
        #test_loader
        for images, labels in test_loader:
            images = images.reshape(-1, 28*28)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum()
        print('Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total))
    torch.save(model.state_dict(), model_weight)

if __name__ == "__main__":
    lgistic_regression()







