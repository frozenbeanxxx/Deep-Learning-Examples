#!/usr/bin/env python3

import os
import re
import sys
import tarfile
import shutil
import math
from functools import reduce
#from six.moves import urllib
import functools
#import pysnooper
#from easydict import EasyDict

def hello_python():
    print("hello python")

    print('The quick brown fox', " jumps over", 'the lazy dog')

    #name = input("please input: ")
    #print(name)

def download_test():
    URL = 'http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz'
    #dest_dir = "D:\\temp\\ccc"
    dest_dir = 'D:/dataset/cifar10/'
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    filename = URL.split('/')[-1]
    print(filename)
    filepath = os.path.join(dest_dir, filename)
    if not os.path.exists(filepath):
        def _progress(count, block_size, total_size):
            sys.stdout.write('\r>> Downloading %s %.1f%%' % (filename,
                float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()
        filepath, _ = urllib.request.urlretrieve(URL, filepath, _progress)
        print()
    statinfo = os.stat(filepath)
    print(statinfo)
    extracted_dir_path = os.path.join(dest_dir, "cifar-10-batches-bin")
    if not os.path.exists(extracted_dir_path):
        tarfile.open(filepath, 'r:gz').extractall(dest_dir)

def class_attribute_test():
    class ClassA(object):
        pass 
        def func(self):
            print("class func")
    a = ClassA()
    a.b = 1.0
    a.c = "s"
    a.d = int(2)
    print(a)
    print(a.b)
    print(a.c)
    print(a.d)
    print('\n'.join(['%s:%s' % item for item in a.__dict__.items()]))
    print(a.__doc__)
    print(a.__class__)
    print(a.__module__)
    print(a.__format__)
    if a.d == 2:
        pass

def string_test():
    temp_str = 'this is a test'
    print(temp_str.replace('is','IS'))
    print(temp_str)
    temp_str = ' this is a test '
    print(temp_str)
    print(temp_str.strip())
    print(temp_str.lstrip())
    print(temp_str.rstrip())
    s1 = 'sssss'
    s2 = 'ddddd'
    s3 = f"{s1}_{s2}"
    print(s3)

def date_type_test():
    # integer float string
    a = 1
    b = 0xff8844
    c = 0o123
    d = 0b101
    print("b %x" % b) # %d %f %s %x
    print("c %d" % c)
    print("d %d" % d)
    print('''first line
... second line
... third line''')
    print(r'\n\t\\')
    print(not True)
    print(True or False)
    print(True and False)
    print(True and False and True and False)
    print(18 > 17)
    e = None 
    print("e : ", e)
    print("PI : ", math.pi)
    print("A : ", ord('A'))
    print("中 : ", ord('中'))
    print("中 : ", '\u4e2d\u6587')
    print('中文'.encode('utf-8'))
    #print('中文'.encode('Unicode')) # LookupError: unknown encoding: Unicode
    print("e : ", chr(0x59))
    print("e : ", str(12345), len("efwe"))

    #!/usr/bin/env python3
    # -*- coding: utf-8 -*-

    print("%d %s %f %x %o " % (123, 123, 123, 123, 123))

def list_test():
    classmates = ['Michael', 'Bob', 'Tracy']
    print(classmates)
    print(len(classmates))
    print(classmates[1])
    print(classmates[-1])
    classmates.append('Adam')
    print(classmates)
    classmates.insert(1, "Jack")
    print(classmates)
    classmates.pop()
    print(classmates)
    classmates.pop(1)
    print(classmates)
    print(classmates)
    print(classmates)
    print(classmates)

def if_test():
    age = 20
    if age >= 6:
        print('teenager')
    elif age >= 18:
        print('adult')
    else:
        print('kid')

    print(range(10))
    print(list(range(10)))

    L = ['Bart', 'Lisa', 'Adam']
    for name in L:
        print('Hello,'+name+'!')
        print('Hello,',name,'!') # 逗号后面会自动

def dict_test():
    c = {'Michael': 96, 'Jimmy': 34}
    d = {'Michael': 95, 'Bob': 75, 'Tracy': 85}
    print(d['Michael'])
    d['Adam'] = 67
    print(d['Adam'])
    print(d.__hash__)
    print(d['Michael'].__hash__)
    print(d['Adam'].__hash__)
    print('Thomas' in d)
    print(d.get('Thomas', -1))
    d.pop('Adam')
    print(d)
    d.update(c)
    print(d)

def set_test():
    a = set([1,2,3])
    print(a)
    a.add(4)
    print(a)
    s = set([5,2,3,3])
    print(s)
    print(a & s)
    print(a | s)
    e = 'abc'
    f = e.replace('a', 'A')
    print(e, f)

def nop():
    pass

def func_test():
    def calc(*nums):
        sum = 0
        for i in nums:
            sum += i 
        print(sum)
    
    a = list(range(10))
    calc(*a)
    calc(1,2,3)

    def person(name, age, **kw):
        print('name:', name, 'age:', age, 'other:', kw)

    person('Adam', 45, gender='M', job='Engineer')
    # func(*args, **kw)

def high_level_feature():
    # 切片
    # 迭代
    from collections import Iterable, Iterator
    print(isinstance('abc', Iterable))
    print(isinstance([1,2,3], Iterable))
    print(isinstance(123, Iterable))
    for i, value in enumerate(['A', 'B', 'C']):
        print(i, value)

    # 列表生成器
    a = list(range(10))
    print(a)
    a = [x * x for x in range(10)]
    print(a)
    a = [x * x for x in range(10) if x % 2 == 0]
    print(a)
    a = [m+n for m in "abc" for n in "xyz"]
    print(a)
    L = ['Hello', 'World', 18, 'Apple', None]
    a = [s.lower() for s in L if isinstance(s, str)]
    print(a)

    # 生成器
    # 返回是个yeild是个生成器

    # 迭代器
    print(isinstance('abc', Iterator))
    print(isinstance([1,2,3], Iterator))
    print(isinstance(123, Iterator))
    print(isinstance(iter('abc'), Iterator))
    print(isinstance(iter([1,2,3]), Iterator))
    #print(isinstance(iter(123), Iterator))

    # 首先获得Iterator对象:
    it = iter([1, 2, 3, 4, 5])
    # 循环:
    while True:
        try:
            # 获得下一个值:
            x = next(it)
        except StopIteration:
            # 遇到StopIteration就退出循环
            break

#@pysnooper.snoop()
def higher_order_function_test():
    # 高阶函数测试
    a = abs(-10)
    print(a)
    a = abs(-14.5555)
    print(a)
    abs_my = abs 
    print(abs_my(-3))
    print(abs, abs_my, abs_my.__doc__)
    def abs_add(a,b,f):
        return f(a) + f(b)
    c = abs_add(-2, -4, abs)
    print(c)
    l = list(range(10))
    print(l)
    def f1(x):
        return x*x 
    d = map(f1, l)
    print(d, list(d))
    d = map(str, l)
    print(d, list(d))
    def f2(x, y):
        return x+y 
    d = reduce(f2, l)
    print(d)

    DIGITS = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9}
    def str2int(s):
        def fn(x, y):
            return x * 10 + y
        def char2num(s):
            return DIGITS[s]
        return reduce(fn, map(char2num, s))
    d = str2int("12345")
    print(d)

    def str2int2(s):
        return reduce(lambda x, y: x*10 + y, map(lambda s : DIGITS[s], s))
    d = str2int("12345")
    print(d)
    l = ["Asd", "Dfg"]
    d = map(lambda s : s.lower(), l)
    print(d, list(d))

    def is_odd(n):
        return n % 2 == 1
    d = list(filter(is_odd, [1, 2, 4, 5, 6, 9, 10, 15]))
    print(d)
    s = " A B "
    print(s.strip())
    print(s and s.strip())

    def not_empty(s):
        return s and s.strip()
    d = list(filter(not_empty, ['A', '', 'B', None, 'C', '  ']))
    print(d)

    def _odd_iter():
        n = 1
        while True:
            n = n + 2
            yield n
    def _not_divisible(n):
        return lambda x: x % n > 0

    def primes():
        yield 2
        it = _odd_iter() # 初始序列
        while True:
            n = next(it) # 返回序列的第一个数
            yield n
            it = filter(_not_divisible(n), it) # 构造新序列
    # 打印1000以内的素数:
    it = _odd_iter()
    d = filter(_not_divisible(13), it)
    for n in d:
        if n < 20:
            print(n)
        else:
            break
    for n in primes():
        if n < 20:
            print(n)
        else:
            break

    l = [36, 5, -12, 9, -21]
    d = sorted(l)
    print(d)
    d = sorted(l, key=abs)
    print(d)
    d = sorted(l, key=lambda x: x if x>0 else -x)
    print(d)
    d = sorted(l, key=abs, reverse=True)
    print(d)

def return_fun_test():
    def calc_sum(*argv):
        ax = 0
        for n in argv:
            ax += n
        return ax 
    l = list(range(10))
    a = calc_sum(*l)
    print(a)

    # 闭包
    def lazy_sum(*argv):
        def sum():
            ax = 0
            for n in argv:
                ax += n
            return ax 
        return sum 
    f = lazy_sum(*l)
    print(f)
    a = f()
    print(a)

def decorator_test():
    def log(func):
        def wrapper(*args, **kw):
            print('call %s():' % func.__name__)
            return func(*args, **kw)
        return wrapper
    @log
    def now():
        print('2015-3-25')
    now()

def partial_function_test():
    # 偏函数测试
    def int2(x, base=2):
        return int(x, base)
    a = int2('10001')
    print(a)
    int22 = functools.partial(int, base=2)
    a = int22('1010')
    print(a)
    max10 = functools.partial(max, 10)
    a = max10(5)
    print(a)

def OOP_test():
    class Student(object):
        def __init__(self, name, age):
            self.name = name 
            self.age = age
            self.__s = 12
        def print_info(self):
            print("name: ", self.name, "age: ", self.age)
    a = Student("jim", 10)
    a.print_info()

    class Animal(object):
        def run(self):
            print('Animal is running...')

    class Cat(Animal):
        pass

    class Dog(Animal):
        def run(self):
            print('Dog is running...')
        def __len__(self):
            return 10

    class Timer(object):
        def run(self):
            print('Timer is running...')

    def run_twice(animal):
        animal.run()
        animal.run()

    run_twice(Animal())
    run_twice(Cat())
    run_twice(Dog())
    run_twice(Timer())
    d = Dog()
    print(isinstance(d, Animal))
    print(isinstance(d, Dog))
    print(isinstance(d, Cat))
    print(dir(d))
    print(dir("ssssdd"))
    print(len(d))
    print(len("ssssdd"))

def return_test():
    print("qqqqqqqqqqqqqq")




#@pysnooper.snoop()
def essydict_test():
    d = EasyDict()
    d.a1 = 3
    d.a2 = 'ww'
    d.a3 = [1,2,3]
    print(d)

def test01():
    a = range(10)
    b = max(a)
    print(b)
    print(math.e)
    print(math.log(2,math.e))
    a = str([1, 2])
    print(a, type(a))
    pro = [0.1, 0.1, 0.1, 0.7]
    for _ in range(100):
        a = np.random.choice(4, 1)#, p=pro)
        #print(a)

def zip_test():
    a = [1,2,3]
    b = [4,5,6]
    c = [4,5,6,7,8]
    zipped = zip(a,b)     # 打包为元组的列表
    print(list(zipped))
    print(zipped)
    print(zip(a,c))              # 元素个数与最短的列表一致

    #print(zip(*zipped))          # 与 zip 相反，*zipped 可理解为解压，返回二维矩阵式
    
    for i in zip(a):
        print(i)
    for i in zip(a,c):
        print(i)
    for i in zip(*zipped):
        print(i)

def print_test():
    for i in range(100):
        print(i, end='\t' if (i+1)%10!=0 else '\n')

def property_test():
    class Student1(object):
        pass
    s = Student1()
    s.score = 9999
    print(s.score)

    class Student2(object):
        def get_score(self):
            return self._score

        def set_score(self, value):
            if not isinstance(value, int):
                raise ValueError('score must be an integer!')
            if value < 0 or value > 100:
                raise ValueError('score must between 0 ~ 100!')
            self._score = value
    s = Student2()
    #s._score = 9998
    s.set_score(91)
    print(s.get_score())
    #print(s._Student2__score)
    s.score = 600
    #s.score = 600
    print(s.score)
    print(s.get_score())
    s._score = 69
    print(s.score)
    print(s.get_score())

    class Student3(object):
        @property
        def score(self):
            return self._score

        @score.setter
        def score(self, value):
            if not isinstance(value, int):
                raise ValueError('score must be an integer!')
            if value < 0 or value > 100:
                raise ValueError('score must between 0 ~ 100!')
            self._score = value
    s = Student3()
    s.score = 60
    #s.score = 600
    print('qqq', s.score)
    s._score = 69
    print(s.score)

def move_file():
    def mymovefile(srcfile,dstfile):
        if not os.path.isfile(srcfile):
            print("%s not exist!"%(srcfile))
        else:
            fpath,fname=os.path.split(dstfile)    #分离文件名和路径
            if not os.path.exists(fpath):
                os.makedirs(fpath)                #创建路径
            shutil.move(srcfile,dstfile)          #移动文件
            print("move %s -> %s"%( srcfile,dstfile))
    def mymovefile2(srcfile,dst_dir):
        if not os.path.isfile(srcfile):
            print("%s not exist!"%(srcfile))
        else:
            fpath,fname=os.path.split(srcfile)
            if not os.path.exists(dst_dir):
                os.makedirs(dst_dir)                #创建路径
            dstfile = os.path.join(dst_dir, fname)
            shutil.move(srcfile,dstfile)          #移动文件
            print("move %s -> %s"%( srcfile,dstfile))
    src_file = "D:\\temp/t2.py"
    dst_dir = "D:\\temp\\aaaaa/"
    mymovefile2(src_file, dst_dir)

def with_test():
    class C_A:
        def __init__(self, msg):
            self.msg = msg
            print('init')
        def __enter__(self):
            print('enter')
        def __exit__(self, exc_type, exc_value, exc_tb):
            print(self.msg %(0.9), exc_type, exc_value, exc_tb)
    with C_A('aaa %f'):
        print('with C_A')

def add_env():
    current_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
    sys.path.append(current_dir)
    print(sys.path)
    print('=================================')
    print(os.environ)

def t1():
    while True:
        s = input('please input something:  ')
        print(s)
        if s == 'q':
            break 

def t2():
    assert 123456 % 32 == 0, 'Multiples of 32 required'
    assert 123456 % 128 == 0, 'Multiples of 128 required'

#download_test()
#class_attribute_test()
#string_test()
#date_type_test()
#list_test()
#if_test()
#dict_test()
#set_test()
#nop()
#func_test()
#high_level_feature()
#higher_order_function_test()
#return_fun_test()
#decorator_test()
#partial_function_test()
#OOP_test()
#return_test()
#essydict_test()
#test01()
#zip_test()
#print_test()
#property_test()
#move_file()

if __name__ == "__main__":
    pass