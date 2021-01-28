import torch 
import torchvision
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms
import cv2 


# ================================================================== #
#                         Table of Contents                          #
# ================================================================== #

# 1. Basic autograd example 1               (Line 25 to 39)
# 2. Basic autograd example 2               (Line 46 to 83)
# 3. Loading data from numpy                (Line 90 to 97)
# 4. Input pipline                          (Line 104 to 129)
# 5. Input pipline for custom dataset       (Line 136 to 156)
# 6. Pretrained model                       (Line 163 to 176)
# 7. Save and load model                    (Line 183 to 189) 


# ================================================================== #
#                     1. Basic autograd example 1                    #
# ================================================================== #
def t1():
    x = torch.tensor(1., requires_grad=True)
    w = torch.tensor(2., requires_grad=True)
    b = torch.tensor(3., requires_grad=True)
    y = w*x + b
    y.backward()
    print(x.grad)
    print(w.grad)
    print(b.grad)

# ================================================================== #
#                    2. Basic autograd example 2                     #
# ================================================================== #
def t2():
    x = torch.randn(10, 3)
    y = torch.randn(10, 2)
    linear = nn.Linear(3,2)
    print('w:', linear.weight)
    print('b:', linear.bias)
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(linear.parameters(), lr=0.01)
    pred = linear(x)
    loss = criterion(pred, y)
    print('loss:', loss.item())
    loss.backward()
    print('dL/dw:', linear.weight.grad)
    print('dL/db:', linear.bias.grad)
    optimizer.step()
    pred = linear(x)
    loss = criterion(pred, y)
    print('loss after 1 step optimization: ', loss.item())

# ================================================================== #
#                     3. Loading data from numpy                     #
# ================================================================== #
def t3():
    x = np.array([[1,2], [3,4]])
    y = torch.from_numpy(x)
    print(y)
    z = y.numpy()
    print(z)

# ================================================================== #
#                         4. Input pipline                           #
# ================================================================== #
def t4():
    cifar10_path = 'E:/dataset/cifar10'
    train_dataset = torchvision.datasets.CIFAR10(root=cifar10_path,
                                                train=True,
                                                transform=transforms.ToTensor(),
                                                download=True)
    image, label = train_dataset[0]
    print(image.size())
    print(label)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                            batch_size=2,
                                            shuffle=True)
    data_iter = iter(train_loader)
    images, labels = data_iter.next()
    print(images)
    print(labels)

    for images, labels in train_loader:
        # Trainign code
        pass

# ================================================================== #
#                5. Input pipline for custom dataset                 #
# ================================================================== #
def t5():
    class CustomDataset(torch.utils.data.Dataset):
        def __init__(self):
            # TODO
            # 1. Initialize file paths or a list of file names.
            pass
        def __getitem__(self, index):
            # TODO
            pass
            # 1. Read one data from files (e.g. using numpy.fromfile, PIL.Image.open).
            # 2.Preprocess the data (e.g. torchvision.Transform).
            # 3.Return a data pair(e.g. image and label)
        def __len__(self):
            # You should change 0 to the total size of your dataset.
            return 0

    class CustomDataset2(torch.utils.data.Dataset):
        def __init__(self):
            ann_file = r'E:\prj\video\stage1\output\annotations\annotations.txt'
            self.file_list = []
            with open(ann_file,"r") as f: 
                for line in f:
                    line = line.strip('\n')
                    self.file_list.append(line)

        def __getitem__(self, index):
            # 1. Read one data from files (e.g. using numpy.fromfile, PIL.Image.open).
            # 2.Preprocess the data (e.g. torchvision.Transform).
            # 3.Return a data pair(e.g. image and label)
            line = self.file_list[index]
            image_path, label = line.split(',')
            img = cv2.imread(image_path)
            label = int(label)
            img = transforms.ToTensor()(img)
            return img, label
        def __len__(self):
            # You should change 0 to the total size of your dataset.
            return len(self.file_list)

    custom_dataset = CustomDataset2()
    train_loader = torch.utils.data.DataLoader(dataset=custom_dataset,
                                                batch_size=2,
                                                shuffle=True)
    data_iter = iter(train_loader)
    images, labels = data_iter.next()
    print(images.shape)
    print(labels)

# ================================================================== #
#                        6. Pretrained model                         #
# ================================================================== #
def t6():
    resnet = torchvision.models.resnet18(pretrained=True)
    for param in resnet.parameters():
        print(param.shape, param.requires_grad)
        param.requires_grad = False 
    resnet.fc = nn.Linear(resnet.fc.in_features, 100)
    images = torch.randn(64, 3, 224, 224)
    outputs = resnet(images)
    print(outputs.size())

# ================================================================== #
#                      7. Save and load the model                    #
# ================================================================== #
def t7():
    resnet = torchvision.models.resnet18(pretrained=True)
    model_path = 'model/202001131519_model.ckpt'
    torch.save(resnet, model_path)
    model = torch.load(model_path)
    print(model)
    weight_path = 'model/202001131519_weight.ckpt'
    torch.save(resnet.state_dict(), weight_path)
    weight = torch.load(weight_path)
    #print(weight)
    resnet.load_state_dict(weight)

if __name__ == "__main__":
    #t1()
    #t2()
    #t3()
    #t4()
    #t5()
    #t6()
    t7()