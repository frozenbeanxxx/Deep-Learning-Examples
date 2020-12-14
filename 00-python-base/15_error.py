import sys

def error_catch_1():
    print("\nerror_catch_1")
    while True:
        try:
            x = int(input("请输入一个数字: "))
            break
        except ValueError:
            print("您输入的不是数字，请再次尝试输入！")

def error_catch_2():
    print("\nerror_catch_2")
    try:
        #f = open('myfile.txt')
        f = open('runoob.txt')
        s = f.readline()
        i = int(s.strip())
    except OSError as err:
        print("OS error: {0}".format(err))
    except ValueError:
        print("Could not convert data to an integer.")
    except:
        print("Unexpected error:", sys.exc_info()[0])
        raise

def error_catch_3():
    print("\nerror_catch_3")
    try:
        s = 'runoob.txt' #'myfile.txt'
        f = open(s)
    except IOError:
        print('cannot open', s)
    else:
        print(s, 'has', len(f.readlines()), 'lines')
        f.close()

def error_catch_4():
    print("\nerror_catch_4")
    try:
        raise NameError('HiThere')
    except NameError:
        print('raise NameError An exception flew by!')

def pre_define_clean():
    print("\npre_define_clean")
    with open('runoob.txt') as f:
        for line in f:
            print(line)

if __name__ == "__main__":
    error_catch_1()
    error_catch_2()
    error_catch_3()
    error_catch_4()
    pre_define_clean()