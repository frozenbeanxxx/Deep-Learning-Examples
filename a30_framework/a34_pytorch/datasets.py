import torchvision.datasets as datasets 

def t1():
    dt = datasets.MNIST('/media/wx/diskE/temp/dt', download=True)
    print(dt)

def t2():
    dt = datasets.Kinetics400('/media/wx/diskE/temp/dt', download=True)
    print(dt)

if __name__ == "__main__":
    t2()