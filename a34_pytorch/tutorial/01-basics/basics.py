import os
import sys
from natsort import natsorted
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
def basic_autograd_example_1():
    print('\n***** run :', sys._getframe().f_code.co_name)
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
def basic_autograd_example_2():
    print('\n***** run :', sys._getframe().f_code.co_name)
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
def loading_data_from_numpy():
    print('\n***** run :', sys._getframe().f_code.co_name)
    x = np.array([[1,2], [3,4]], dtype=np.float64)
    y = torch.from_numpy(x)
    print(y)
    z = y.numpy()
    print(z)

# ================================================================== #
#                         4. Input pipline                           #
# ================================================================== #
def input_pipline():
    print('\n***** run :', sys._getframe().f_code.co_name)
    cifar10_path = '/media/weixing/diskD/dataset/pytorch/cifar10'
    if not os.path.exists(cifar10_path):
        print(cifar10_path, 'not exists')
        return None
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
    #print(images)
    print('sample image size:', images.size())
    print('labels size:', labels.size(), 'label:', labels)

    for images, labels in train_loader:
        # Trainign code
        pass

# ================================================================== #
#                5. Input pipline for custom dataset                 #
# ================================================================== #
def input_pipline_for_custom_dataset():
    print('\n***** run :', sys._getframe().f_code.co_name)
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

    class CustomDataset3(torch.utils.data.Dataset):
        def __init__(self, root_dir):
            self.root_dir = root_dir
            image_dirs = os.listdir(self.root_dir)
            image_dirs = natsorted(image_dirs)
            self.label_dict = {}
            self.file_list = []
            for i, category in enumerate(image_dirs):
                self.label_dict[category] = i
                images_dir = os.path.join(self.root_dir, category)
                tmp_file_list = os.listdir(images_dir)
                for filename in tmp_file_list:
                    if filename[-4:] == '.jpg' or filename[-4:] == '.png':
                        self.file_list.append(os.path.join(category, filename))
            self.classes_num = len(image_dirs)
        def __getitem__(self, index):
            filepath = self.file_list[index]
            category = filepath.split('/')[0]
            label = self.label_dict[category]
            image_path = os.path.join(self.root_dir, filepath)
            img = cv2.imread(image_path)
            img = transforms.ToTensor()(img)
            return img, label
        def __len__(self):
            # You should change 0 to the total size of your dataset.
            return len(self.file_list)

    image_rootdir = '/media/weixing/diskD/dataset/other/CatAndDog/PetImages'
    custom_dataset = CustomDataset3(image_rootdir)
    train_loader = torch.utils.data.DataLoader(dataset=custom_dataset,
                                                batch_size=1,
                                                shuffle=True)
    data_iter = iter(train_loader)
    images, labels = data_iter.next()
    print(images.shape)
    print(labels)

# ================================================================== #
#                        6. Pretrained model                         #
# ================================================================== #
def pretrained_model():
    print('\n***** run :', sys._getframe().f_code.co_name)
    resnet = torchvision.models.resnet18(pretrained=True)
    for param in resnet.parameters():
        #print(param.shape, param.requires_grad)
        param.requires_grad = False 
    resnet.fc = nn.Linear(resnet.fc.in_features, 100)
    images = torch.randn(64, 3, 224, 224)
    outputs = resnet(images)
    print(outputs.size())

# ================================================================== #
#                      7. Save and load the model                    #
# ================================================================== #
def save_and_load_model():
    resnet = torchvision.models.resnet18(pretrained=True)
    model_dir = 'model'
    os.makedirs(model_dir, exist_ok=True)
    model_path = model_dir + '/model.ckpt'
    torch.save(resnet, model_path)
    model = torch.load(model_path)
    print(model)
    weight_path = model_dir + '/weight.ckpt'
    torch.save(resnet.state_dict(), weight_path)
    weight = torch.load(weight_path)
    #print(weight)
    resnet.load_state_dict(weight)

if __name__ == "__main__":
    basic_autograd_example_1()
    basic_autograd_example_2()
    loading_data_from_numpy()
    #input_pipline()
    input_pipline_for_custom_dataset()
    pretrained_model()
    save_and_load_model()