import numpy as np 
import fire 
import time 
from matplotlib import pyplot as plt

def test1():
    #np.random.seed(10)
    np.random.seed(int(time.time()))
    a = np.random.randint(10, size=10)
    print('random', a)
    a = np.random.randint(10, size=(10, 5))
    print('random', a)
    a = np.full((2,3), 10)
    a[0] = [1, 2, 3]
    print(a)

def test2():
    a = np.arange(0, 24)
    print(a)
    a = a.reshape(2, -1)
    print(a)
    a = a.reshape(3, 2, -1)
    print(a)
    a = a.reshape(-1, 2, 3)
    print(a)

def BaseTest1():
    print('create numpy array')
    a = np.array([1,2,3,4,5])
    print(a)
    print(a.shape)
    print(a[0], a[1])
    a[0] = -1
    print(a)
    a = np.zeros((5))
    print(a)
    a = np.ones((5))
    print(a)
    a = np.random.random((5)) # between 0~1
    print(a)
    a = np.zeros((2,3))
    print(a)
    a = np.ones((2,4))
    print(a)
    a = np.array([[4,5], [6,1]])
    print(a[0][1])
    b = a[:,1]
    print(b)
    c = np.full((5,2), 123)
    print('\ninitial with a special number\n', c)

def BaseTest2():
    a = np.array([[1.0, 2.0], [3.0, 4.0]]) 
    b = np.array([[5.0, 6.0], [7.0, 8.0]]) 
    sum = a + b 
    difference = a - b 
    product = a * b 
    quotient = a / b 
    print("Sum = \n", sum )
    print("Difference = \n", difference )
    print("Product = \n", product )
    print("Quotient = \n", quotient )
    matrix_product = a.dot(b) 
    print("Matrix Product = \n", matrix_product)
    #x1 = np.array([[1.0, 2.0], [3.0, 4.0]]) 
    #x2 = np.array([[3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]) 
    #y = x1 * x2 
    #print(y)
    a = np.array(b, dtype=complex, ndmin=4) 
    print(a)
    dt = np.dtype(np.int32)
    print(dt)
    dt = np.dtype('i4')
    print(dt)
    dt = np.dtype('<i4')
    print(dt)
    dt = np.dtype([('age', np.int8)])
    print(dt)
    a = np.array([(10,), (20,), (30,)], dtype=dt)
    print(a, type(a))
    print(a['age'])
    student = np.dtype([('name','S20'), ('age', 'i1'), ('marks', 'f4')]) 
    print(student)
    a = np.array([('abc', 21, 50),('xyz', 18, 75)], dtype = student) 
    print(a)
    print(a['name'])
    print(a['age'])
    print('%i, %i' % (True, 10))

    a = np.arange(24)  
    print (a.ndim)             # a 现只有一个维度
    # 现在调整其大小
    b = a.reshape(2,4,3)  # b 现在拥有三个维度
    print (b.ndim)

    # 数组的 dtype 为 int8（一个字节）  
    x = np.array([1,2,3,4,5], dtype = np.int8)  
    print (x.itemsize)
    
    # 数组的 dtype 现在为 float64（八个字节） 
    y = np.array([1,2,3,4,5], dtype = np.float64)  
    print (y.itemsize)

    x = np.array([1,2,3,4,5])  
    print (x.flags) 
    b = x
    print(b.flags)

    s =  b'Hello World' 
    a = np.frombuffer(s, dtype =  'S1')  
    print (a)

    list=range(5)
    it=iter(list)
    # 使用迭代器创建 ndarray 
    x=np.fromiter(it, dtype=float)
    print(x)

def BaseTest3():
    # 同类型多维数组，一张表，正整数元祖索引，维度称轴，轴的数目为rank
    # numpy数组类ndarray
    a = np.arange(15).reshape(3,5)
    print(a)
    print(a.shape)
    print(a.ndim)
    print(a.dtype, a.dtype.name)
    print(a.size)
    print(type(a))

    a = np.array([1.2,3.5,5.1])
    print(a)
    a = np.array([(1.5,2,3), (4,5,6)])
    print(a)
    a = np.array( [ [1,2], [3,4] ], dtype=complex )
    print(a)
    a = np.zeros((3,4))
    print(a)
    a = np.ones((2,3,4), dtype=np.int16)
    print(a)
    a = np.empty((2,3))
    print(a, a.dtype)
    a = np.arange(0,2,0.3)
    print(a)
    a = np.linspace(0,2*np.pi, 9)
    print(a)
    a = np.arange(1000).reshape(10,-1)
    #np.set_printoptions(threshold=np.inf)
    np.set_printoptions(threshold=100)
    print(a)
    a = np.logspace(0,9,10,base=2)
    print ('logspace', a)

    a = np.arange(12).reshape(3,4)
    b = np.random.randint(0,100,12).reshape(3,4)
    print(a)
    print(b)
    c = a - b 
    print(c)
    c = a**2
    print(c)
    c = 10*np.sin(a)
    print(c)
    c = a < b 
    print(c)
    d = np.transpose(b)
    print(d)
    c = a.dot(d)
    print(c)
    c = np.dot(a,d)
    print(c)
    c = a.sum() # exp, sqrt, sin, cos, add
    print(c)
    c = a.cumsum()
    print(c)
    c = a.min()
    print(c)
    c = a.max()
    print(c)
    c = a.sum(axis=1)
    print(c)
    c = a.cumsum(axis=0)
    print(c)
    c = a.min(axis=1)
    print(c)
    c = a.max(axis=0)
    print(c)

    a = np.arange(10)**3
    print(a)
    print(a[2])
    print(a[2:5])
    print(a[:6:2])
    print(a[::-1])
    for i in a:
        print(i**(1/3.))

    def f(x,y):
        return 10*x+y 
    a = np.fromfunction(f, (5,4), dtype=int)
    print(a)
    print(a[-1])
    print(a[-1,...]) # ... means other axis
    for row in a:
        print(row)
    for element in a.flat:
        print(element)

    x = np.array([[  0,  1,  2],[  3,  4,  5],[  6,  7,  8],[  9,  10,  11]])  
    print ('我们的数组是：')
    print (x)
    print ('\n')
    # 现在我们会打印出大于 5 的元素  
    print  ('大于 5 的元素是：')
    print (x[x >  5])
    a = np.array([np.nan,  1,2,np.nan,3,4,5])  
    print (a[~np.isnan(a)])
    a = np.array([1,  2+6j,  5,  3.5+5j])  
    print (a[np.iscomplex(a)])
    x=np.arange(32).reshape((8,4))
    print (x[np.ix_([1,5,7,2],[0,3,1,2])]) # 按照维度顺序依次索引

    a = np.array([[ 0, 0, 0],
           [10,10,10],
           [20,20,20],
           [30,30,30]])
    b = np.array([1,2,3])
    print(a + b)
    print(a * b)
    bb = np.tile(b, (4, 1))
    print(bb)
    #c = np.array([[1,2,3], [1,2,3]])
    #print(a + c)

def ShapeOperator():
    a = np.floor(10*np.random.random((3,4)))
    print(a)
    print(a.shape)
    print(a.ravel())
    print(a.reshape(6,2)) # a.resize是修改a本身
    print(a.T)
    print(a.T.shape)

    a = np.floor(10*np.random.random((2,2)))
    print(a)
    b = np.floor(10*np.random.random((2,2)))
    print(b)
    c = np.vstack((a,b))
    print(c)
    c = np.hstack((a,b))
    print(c)
    c = np.column_stack((a,b))
    print(c)
    a = np.array([4.,2.])
    b = np.array([3.,8.])
    c = np.column_stack((a,b))
    print(c)
    c = np.hstack((a,b))
    print(c)
    c = a[:,np.newaxis]
    print(c)
    c = np.r_[1:4,0,4]
    print(c)
    #c = np.c_[1:4,0,4] # ValueError: all the input array dimensions except for the concatenation axis must match exactly
    #print(c)
    c = np.concatenate((a,b), axis=0)
    print(c)

    # hspilit沿横轴分割，vsplit沿纵轴分割，array_split沿指定轴分割
    a = np.floor(10*np.random.random((2,12)))
    print(a)
    c = np.hsplit(a,3)
    print(c)
    c = np.hsplit(a,(3,4))
    print(c)

def CopyTest():
    a = np.arange(12)
    print(a)
    b = a
    print(b is a)
    b.shape = 3,4
    print(a)
    print(a.shape)
    print(id(a))
    def f(x):
        print(id(x))
    f(a)

    c = a.view()
    print(c is a)
    print(c.base is a)
    print(c.flags.owndata)
    c.shape = 2,6
    print(c.shape)
    print(a.shape)
    c[0,4] = 1234
    print(c)
    print(a)

    s = a[:,1:3]
    s[:] = 10
    print(a)

    # deep copy
    d = a.copy()
    d[0,0] = 9999
    print(a)
    print(d)

'''
usually use functions
1. create array
arange, array, copy, empty, empty_like, eye, fromfile, fromfunction,
identity, linspace, logspace, mgrid, ogrid, ones, ones_like, zeros, zeros_like

2. convert
ndarray, astype, atleast_1d, atleast_2d, atleast_3d, mat

3. technique
array_split, column_stack, concatenate, diagonal, dsplit, dstack, hsplit, hstack,
ndarray.item, newaxis, ravel, repeat, reshape, resize, squeeze, swapaxes, take, 
transpose, vsplit, vstack

4. problem
all, any, nonzero, where

5. sort
argmax, argmin, argsort, max, min, ptp, searchsorted, sort

6. operator
choose, compress, cumprod, cumsum, inner, ndarray.fill, imag, prod, put,
putmask, real, sum

7. statistics
cov, mean, std, var

8. base linear algebra
cross, dot, outer, linalg.svd, vdot
linalg means linear algorithm
'''

def IterationTest():
    a = np.arange(6).reshape(2,3)
    print ('原始数组是：')
    print (a)
    print ('\n')
    print ('迭代输出元素：')
    for x in np.nditer(a):
        print (x, end=", " )
    print ('\n')

    a = np.arange(6).reshape(2,3)
    for x in np.nditer(a.T):
        print (x, end=", " )
    print ('\n')
    
    for x in np.nditer(a.T.copy(order='C')):
        print (x, end=", " )
    print ('\n')

    a = np.arange(0,60,5) 
    a = a.reshape(3,4)  
    print ('原始数组是：') 
    print (a) 
    print ('\n') 
    print ('原始数组的转置是：') 
    b = a.T 
    print (b) 
    print ('\n') 
    print ('以 C 风格顺序排序：') 
    c = b.copy(order='C')  
    print (c)
    for x in np.nditer(c):  
        print (x, end=", " )
    print  ('\n') 
    print  ('以 F 风格顺序排序：')
    c = b.copy(order='F')  
    print (c)
    for x in np.nditer(c):  
        print (x, end=", " )
    print  ('\n') 

    a = np.arange(0,60,5) 
    a = a.reshape(3,4)  
    print ('原始数组是：')
    print (a)
    print ('\n')
    print ('以 C 风格顺序排序：')
    for x in np.nditer(a, order =  'C'):  
        print (x, end=", " )
    print ('\n')
    print ('以 F 风格顺序排序：')
    for x in np.nditer(a, order =  'F'):  
        print (x, end=", " )
    print  ('\n') 

    a = np.arange(0,60,5) 
    a = a.reshape(3,4)  
    print ('原始数组是：')
    print (a)
    print ('\n')
    for x in np.nditer(a, op_flags=['readwrite']): 
        x[...]=2*x 
    print ('修改后的数组是：')
    print (a)

    a = np.arange(0,60,5) 
    a = a.reshape(3,4)  
    print ('原始数组是：')
    print (a)
    print ('\n')
    print ('修改后的数组是：')
    for x in np.nditer(a, flags =  ['external_loop'], order =  'F'):  
        print (x, end=", " )
    print  ('\n') 

def ArrayOperatorTest():# difficult
    # 创建了三维的 ndarray
    a = np.arange(8).reshape(2,2,2)
    
    print ('原数组：')
    print (a)
    print ('\n')
    # 将轴 2 滚动到轴 0（宽度到深度）
    
    print ('调用 rollaxis 函数：')
    print (np.rollaxis(a,2))
    # 将轴 0 滚动到轴 1：（宽度到高度）
    print ('\n')
    
    print ('调用 rollaxis 函数：')
    print (np.rollaxis(a,2,1))
    print ('\n')
    print (np.rollaxis(a,1))

    # 创建了三维的 ndarray
    a = np.arange(8).reshape(2,2,2)
    
    print ('原数组：')
    print (a)
    print ('\n')
    # 现在交换轴 0（深度方向）到轴 2（宽度方向）
    
    print ('调用 swapaxes 函数后的数组：')
    print (np.swapaxes(a, 2, 0))

    a = np.array([[1,2,3],[4,5,6]])
    print ('第一个数组：')
    print (a)
    print ('\n')
    
    print ('第一个数组的形状：')
    print (a.shape)
    print ('\n')
    b = np.resize(a, (3,2))
    
    print ('第二个数组：')
    print (b)
    print ('\n')
    
    print ('第二个数组的形状：')
    print (b.shape)
    print ('\n')
    # 要注意 a 的第一行在 b 中重复出现，因为尺寸变大了
    
    print ('修改第二个数组的大小：')
    b = np.resize(a,(3,5))
    print (b)

    # resize, append, insert, delete, unique

def BitOperatorTest():
    # bitwise_and, bitwise_or, invert, left_shift, right_shift
    print(np.binary_repr(40, width=8))
    print(bin(40))
    #print(bin(40.1)) # error

def StatisticsTest():
    a = np.array([[3,7,5],[8,4,3],[2,4,9]])  
    print ('我们的数组是：')
    print (a)
    print ('\n')
    print ('调用 amin() 函数：')
    print (np.amin(a,1))
    print ('\n')
    print ('再次调用 amin() 函数：')
    print (np.amin(a,0))
    print ('\n')
    print ('调用 amax() 函数：')
    print (np.amax(a))
    print ('\n')
    print ('再次调用 amax() 函数：')
    print (np.amax(a, axis =  0))

    a = np.array([[3,7,5], [8,4,3], [2,4,9]])
    print(a)
    print(np.ptp(a))
    print(np.ptp(a, axis=1))
    print(np.ptp(a, axis=0))

    # percentile, median, mean, average, std, var

def SortTest():
    a = np.random.randint(0,100,32)
    print(a)
    print(np.sort(a))
    print(np.sort(a, kind='heapsort'))
    print(np.argsort(a))
    print(np.argsort(a, kind='heapsort'))
    # lexsort, msort, sort_complex, partition, argpartition
    # argmax, argmin
    a = np.array([[30,40,0],[0,20,10],[50,0,60]])  
    print ('我们的数组是：')
    print (a)
    print ('\n')
    print ('调用 nonzero() 函数：')
    b = np.nonzero(a)
    print (np.nonzero (a))
    print(b)

    x = np.arange(9.).reshape(3,  3)  
    print ('我们的数组是：')
    print (x)
    print(x[(0,1), (0,1)])
    print ( '大于 3 的元素的索引：')
    y = np.where(x >  3)  
    print (y)
    print ('使用这些索引来获取满足条件的元素：')
    print (x[y])

    x = np.arange(9.).reshape(3,  3)  
    print ('我们的数组是：')
    print (x)
    # 定义条件, 选择偶数元素
    condition = np.mod(x,2)  ==  0  
    print ('按元素的条件值：')
    print (condition)
    print ('使用条件提取元素：')
    print (np.extract(condition, x))

def ByteswapTest():
    a = np.array([1,  256, 4096,  8755], dtype = np.int16)  
    print ('我们的数组是：')
    print (a)
    print ('以十六进制表示内存中的数据：')
    print (map(hex,a))
    # byteswap() 函数通过传入 true 来原地交换 
    print ('调用 byteswap() 函数：')
    print (a.byteswap(True))
    print ('十六进制形式：')
    print (map(hex,a))
    print (hex(1))
    print (hex(256))

def SaveTest():
    a = np.array([1,2,3,4,5]) 
 
    # 保存到 outfile.npy 文件上
    np.save('outfile.npy',a) 
    
    # 保存到 outfile2.npy 文件上，如果文件路径末尾没有扩展名 .npy，该扩展名会被自动加上
    np.save('outfile2',a)

    b = np.load('outfile.npy')  
    print (b)

    a = np.array([[1,2,3],[4,5,6]])
    b = np.arange(0, 1.0, 0.1)
    c = np.sin(b)
    # c 使用了关键字参数 sin_array
    np.savez("runoob.npz", a, b, sin_array = c)
    r = np.load("runoob.npz")  
    print(r.files) # 查看各个数组名称
    print(r["arr_0"]) # 数组 a
    print(r["arr_1"]) # 数组 b
    print(r["sin_array"]) # 数组 c

    a = np.array([1,2,3,4,5]) 
    np.savetxt('out.txt',a) 
    b = np.loadtxt('out.txt') 
    print(b)

def MatplotlibTest():
    x = np.arange(1,11) 
    y =  2  * x +  5 
    plt.title("Matplotlib demo") 
    plt.xlabel("x axis caption") 
    plt.ylabel("y axis caption") 
    plt.plot(x,y) 
    plt.show()



#test1()
#test2()
#BaseTest1()
#BaseTest2()

# 基础知识、形状操作、复制和视图、深拷贝、Less基础、线性代数、
#BaseTest3()
#ShapeOperator()
#CopyTest()
#IterationTest()
#ArrayOperatorTest()
#BitOperatorTest()
#StatisticsTest()
#SortTest()
#ByteswapTest()
#SaveTest()
#MatplotlibTest()

if __name__ == "__main__":
    fire.Fire()