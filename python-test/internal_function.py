import fire 
import numpy as np

'''
abs()	dict()	help()	min()	setattr()
all()	dir()	hex()	next()	slice()
any()	divmod()	id()	object()	sorted()
ascii()	enumerate()	input()	oct()	staticmethod()
bin()	eval()	int()	open()	str()
bool()	exec()	isinstance()	ord()	sum()
bytearray()	filter()	issubclass()	pow()	super()
bytes()	float()	iter()	print()	tuple()
callable()	format()	len()	property()	type()
chr()	frozenset()	list()	range()	vars()
classmethod()	getattr()	locals()	repr()	zip()
compile()	globals()	map()	reversed()	__import__()
complex()	hasattr()	max()	round()	 
delattr()	hash()	memoryview()	set()
'''

def t1():
    print("========== abs ==========")
    print ("abs(-40) : ", abs(-40))
    print ("abs(100.10) : ", abs(100.10))

    print("========== all ==========")
    print("all(['a', 'b', 'c']):", all(['a', 'b', 'c']))
    print("all(['0', '1', '2']):", all(['0', '1', '2']))
    print("all([0, 1, 2]):", all([0, 1, 2]))
    print("all(('a', 'b', 'c')):", all(('a', 'b', 'c')))
    print("all((0, 1, 2)):", all((0, 1, 2)))
    print("all([]):", all([]))
    print("all(()):", all(()))
    a = np.array([0,1,2])
    print(all(a))

    print("========== any ==========")
    print(any(['a', 'b', 'c', 'd']))  # 列表list，元素都不为空或0
    print(any(['a', 'b', '', 'd']))   # 列表list，存在一个为空的元素
    print(any([0, '', False]))        # 列表list,元素全为0,'',false
    print(any(('a', 'b', 'c', 'd')))  # 元组tuple，元素都不为空或0
    print(any(('a', 'b', '', 'd')))   # 元组tuple，存在一个为空的元素
    print(any((0, '', False)))        # 元组tuple，元素全为0,'',false
    print(any([])) # 空列表
    print(any(())) # 空元组

    print("========== basestring ==========")
    a = isinstance("ssss", str)
    print(a)
    # a = isinstance("ssss", basestring) # in python3, basestring is deleted
    # print(a)

    print("========== ascii ==========")
    a = ascii('aaaa')
    print(a)
    a = ascii('安拉胡克')
    print(a)
    print('\u5b89')

    print("========== bin ==========")
    a = 10
    print(bin(a))
    a = 10000000000000
    print(bin(a))

    print("========== bin ==========")
    print(bool())
    print(bool(0))
    print(bool(1))
    print(issubclass(bool, int))

    print("========== bytearray ==========")
    print(bytearray())
    print(bytearray([1,2,3]))
    print(bytearray('eeee', 'utf-8'))

    print("========== bytes ==========")
    a =bytes([1,2,3,4])
    print(a, type(a))
    a = bytes('hello', 'ascii')
    print(a, type(a))

    print("========== callable ==========")
    print(callable(0))
    def add(a,b):
        return a+b
    print(callable(add))
    class A:
        def fun():
            return 0
    print(callable(A))
    a = A()
    print(callable(a))
    class B:
        def __call__(self):
            return 0
    print(callable(B))
    b = B()
    print(callable(b))

    print("========== chr ==========")
    print(chr(0x30))
    # 0~1114111/0x10FFFF
    for i in range(2048*8): 
        #print(chr(i),bytes(chr(i), 'ascii'), end='')
        print(chr(i), end='')
    print()

    print("========== classmethod ==========")
    class C(object):
        bar = 1
        def func1(self):
            print('foo')
        @classmethod
        def func2(cls):
            print('func2')
            print(cls.bar)
            cls().func1()
    C.func2()

    print("========== compile ==========")
    s = "for i in range(10): print(i)"
    c = compile(s, '', 'exec')
    print(c)
    exec(c)
    s2 = "3*4+5"
    a2 = compile(s2, '', 'eval')
    print(eval(a2))

    print("========== complex ==========")
    print(complex(1,2))
    print(complex(1))
    print(complex("1"))
    #print(complex("1234", "333"))
    print(complex("1+2j"))

    print("========== bytearray ==========")
    print("========== bytearray ==========")
    print("========== bytearray ==========")
    print("========== bytearray ==========")
    print("========== bytearray ==========")
    print("========== bytearray ==========")
    print("========== bytearray ==========")
    print("========== bytearray ==========")
    print("========== bytearray ==========")
    print("========== bytearray ==========")
    print("========== bytearray ==========")
    print("========== bytearray ==========")
    print("========== bytearray ==========")
    print("========== bytearray ==========")
    print("========== bytearray ==========")
    print("========== bytearray ==========")
    print("========== bytearray ==========")
    print("========== bytearray ==========")
    print("========== bytearray ==========")
    print("========== bytearray ==========")
    print("========== bytearray ==========")
    # print("========== dict ==========")
    # print("dict():", dict())
    # print("dict(a='a', b='b', t='t'):", dict(a='a', b='b', t='t'))
    # print("dict(zip(['one', 'two', 'three'], [1, 2, 3])):", dict(zip(['one', 'two', 'three'], [1, 2, 3])))
    # print("dict([('one', 1), ('two', 2), ('three', 3)]):", dict([('one', 1), ('two', 2), ('three', 3)]))
    # print("dict([(1, 'one'), (2, 'two')]):", dict([(1, 'one'), (2, 'two')]))




if __name__ == "__main__":
    fire.Fire()