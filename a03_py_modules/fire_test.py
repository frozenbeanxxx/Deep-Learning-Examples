import fire 
from typing import Callable, List, Union, Optional

class Example(object):
    def hello(self, name='world'):
        return 'Hello {name}!'.format(name=name)

def hello(name='world'):
    return 'Hello {name}!'.format(name=name)

def test1():
    print('test1')

def add(x, y):
    return x + y

def multiply(x, y):
    return x * y

class C1(object):
    def __init__(self):
        print('entry C1 init')
    def __call__(self, config, v, num: Optional[int]=None):\
        print(config)

if __name__ == "__main__":
    a = C1()
    fire.Fire()
    #fire.Fire(hello)
    #fire.Fire({
    #    'add':add,
    #    'multiply':multiply,
    #})