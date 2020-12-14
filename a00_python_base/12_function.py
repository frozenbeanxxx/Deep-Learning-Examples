def function_1(s):
    s += "tail"
    print("in  func", s)

def function_2(a):
    print("%X" % id(a))   # 指向的是同一个对象
    a=10
    print("%X" % id(a))   # 一个新对象

def function_3( mylist ):
    "修改传入的列表"
    mylist.append([1,2,3,4])
    print ("函数内取值: ", mylist)
    return

def func_param_keyword( name, age ):
    "打印任何传入的字符串"
    print ("名字: ", name)
    print ("年龄: ", age)
    return

def func_param_default( name, age = 35 ):
    "打印任何传入的字符串"
    print ("名字: ", name)
    print ("年龄: ", age)
    return

def func_param_variable( arg1, *vartuple ):
    "打印任何传入的参数"
    print ("输出: ")
    print (arg1)
    print (vartuple)

def func_param_variable_method_2( arg1, **vardict ):
    "打印任何传入的参数"
    print ("输出: ")
    print (arg1)
    print (vardict)

def function_4(a, b, *, c):
    return a+b+c

def function_5(a, b, *w, c):
    sum = 0
    for i in w:
        sum += i
    return a+b+c+sum

def func_lambda():
    print("\nlambda")
    print("lambda [arg1 [,arg2,......, argn]]:expression")
    sum = lambda arg1, arg2: arg1 + arg2
    print("sum = lambda arg1, arg2: arg1 + arg2")
    print ("10, 20 相加后 sum( 10, 20 ) 的值为 : ", sum( 10, 20 ))
    print ("20, 20 相加后 sum( 20, 20 ) 的值为 : ", sum( 20, 20 ))

def func_force_position(a, b, /, c, d, *, e, f):
    print("\n/之前必须使用指定位置传参")
    print(a, b, c, d, e, f)

if __name__ == "__main__":
    s = 'this is a string.'
    function_1(s)
    print("out func", s)
 
    a=1
    print("%X" % id(a))
    function_2(a)

    mylist = [10,20,30]
    function_3( mylist )
    print ("函数外取值: ", mylist)

    print()
    print("必需参数")
    print("关键字参数")
    func_param_keyword(age=50, name="runoob")
    print("默认参数")
    func_param_default(age=50, name="runoob")
    func_param_default(name="runoob" )
    print("不定长参数, * is tuple, ** is dictionary")
    func_param_variable(10, 20, 30)
    func_param_variable_method_2(10, a=20, b=30)
    print("function_4", function_4(1, 2, c=4))
    print("function_5", function_5(1, 2, 5, 5, 5, c=4))

    func_lambda()

    func_force_position(10, 20, 30, d=40, e=50, f=60)