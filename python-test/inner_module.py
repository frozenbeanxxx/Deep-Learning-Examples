from collections import namedtuple

def t1():
    Point = namedtuple('Point', ['x', 'y'])
    p = Point(1, 2)
    print(p)
    #print(p.__dict__)
    print(p.__doc__)
    print(isinstance(p, Point))
    print(isinstance(p, tuple))

class C1(namedtuple('C1', ['x', 'y'])):
    def __new__(cls, _x, _y):
        x = _x
        y = _y
        return super().__new__(cls, x, y)

def t2():
    c = C1(1,2)
    print(c)

if __name__ == "__main__":
    t2()