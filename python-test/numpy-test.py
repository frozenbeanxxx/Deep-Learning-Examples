import numpy as np 
import time 

def test1():
    #np.random.seed(10)
    np.random.seed(int(time.time()))
    a = np.random.randint(10, size=10)
    print('random', a)
    a = np.random.randint(10, size=(10, 5))
    print('random', a)

def test2():
    a = np.arange(0, 24)
    print(a)
    a = a.reshape(2, -1)
    print(a)
    a = a.reshape(3, 2, -1)
    print(a)
    a = a.reshape(-1, 2, 3)
    print(a)

#test1()
test2()