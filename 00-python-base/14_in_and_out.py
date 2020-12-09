import pickle
import pprint

def print_info():
    s = "runoob"
    print("s:", s)
    print("repr(s):", repr(s))

    x = 10 * 3.25
    y = 200 * 200
    s = 'x 的值为： ' + repr(x) + ',  y 的值为：' + repr(y) + '...'
    print(s)

    print("\nprint a square cubic table method 1")
    for x in range(1, 11):
        print(repr(x).rjust(2), repr(x*x).rjust(3), repr(x**3).rjust(4))

    print("\nprint a square cubic table method 2")
    for x in range(1, 11):
        print('{0:2d} {1:3d} {2:8d}'.format(x, x**2, x**3))

def input_something():
    print("\ninput_something")
    s = input("please input:")
    print("s:", s)

def operate_file():
    print("\noperate_file")

    f = open('test.txt', 'w')
    f.write( "Python 是一个非常好的语言。\n是的，的确非常好!!\n" )
    f.close()

    f = open("test.txt", "r")
    s = f.read()
    print(s)
    f.close()

    f = open("test.txt", "r")
    s = f.readline()
    print(s)
    print(repr(s))
    f.close()

    f = open("test.txt", "r")
    s = f.readlines()
    print(s)
    f.close()

def use_pickle():
    print("\nuse_pickle")

    # 使用pickle模块将数据对象保存到文件
    data1 = {'a': [1, 2.0, 3, 4+6j],
            'b': ('string', u'Unicode string'),
            'c': None}

    selfref_list = [1, 2, 3]
    selfref_list.append(selfref_list)

    output = open('data.pkl', 'wb')

    # Pickle dictionary using protocol 0.
    pickle.dump(data1, output)

    # Pickle the list using the highest protocol available.
    pickle.dump(selfref_list, output, -1)

    output.close()

    #使用pickle模块从文件中重构python对象
    pkl_file = open('data.pkl', 'rb')

    print("pkl:", pkl_file)

    data1 = pickle.load(pkl_file)
    pprint.pprint(data1)

    data2 = pickle.load(pkl_file)
    pprint.pprint(data2)

    pkl_file.close()

def use_pprint():
    print("\npprint is more easy to read")

    a = list(dir())
    #pprint.pprint("\nglobal dir(): ", a)
    stuff = ['spam', 'eggs', 'lumberjack', 'knights', 'ni']
    stuff.insert(0, stuff[:])
    #class pprint.PrettyPrinter(indent=1, width=80, depth=None, stream=None, *, compact=False)
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(stuff)
    pp.pprint(a)

if __name__ == "__main__":
    print_info()
    #input_something()
    operate_file()
    use_pickle()