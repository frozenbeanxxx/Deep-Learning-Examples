import sys
from sys import flags

print("\nglobal dir(): ", dir())

def import_1():
    print('命令行参数如下:')
    for i in sys.argv:
        print(i)
    print('\n\nPython 路径为：', sys.path, '\n')
    print("sys.__name__: ", sys.__name__)
    print("\ndir(sys): ", dir(sys))
    print("\ndir(): ", dir())
    print(flags)


if __name__ == "__main__":
    import_1()