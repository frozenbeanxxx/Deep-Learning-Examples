import torch
import torchvision.models as models

from torchsummary import summary

def t1():
    print('torch.cuda.is_available()', torch.cuda.is_available())
    model = models.inception_v3(pretrained=True)
    #print(model)
    print(model.__doc__)
    #print(model.__dict__)
    #print(model.parameters)
    summary(model, (3,512,512), device='cpu')

if __name__ == "__main__":
    t1()