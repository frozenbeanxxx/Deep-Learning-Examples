import fire 

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

if __name__ == "__main__":
    #fire.Fire()
    #fire.Fire(hello)
    fire.Fire({
        'add':add,
        'multiply':multiply,
    })