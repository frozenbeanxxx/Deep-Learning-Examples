import sys

def common_func():
    print('\n Test: a common function')
    def hi(name="yasoob"):
        return "hi " + name
    print(hi())
    # output: 'hi yasoob'
    # 我们甚至可以将一个函数赋值给一个变量，比如
    greet = hi
    # 我们这里没有在使用小括号，因为我们并不是在调用hi函数
    # 而是在将它放在greet变量里头。我们尝试运行下这个
    print(greet())
    # output: 'hi yasoob'
    # 如果我们删掉旧的hi函数，看看会发生什么！
    del hi
    try:
        print(hi())
    except UnboundLocalError as err:
        print(sys.exc_info()[0], err)
    #outputs: NameError
    print(greet())
    #outputs: 'hi yasoob'

def func_in_func():
    print('\n Test: function in function')
    def hi(name="yasoob"):
        print("now you are inside the hi() function")
    
        def greet():
            return "now you are in the greet() function"
    
        def welcome():
            return "now you are in the welcome() function"
    
        print(greet())
        print(welcome())
        print("now you are back in the hi() function")
    
    hi()
    #output:now you are inside the hi() function
    #       now you are in the greet() function
    #       now you are in the welcome() function
    #       now you are back in the hi() function
    
    # 上面展示了无论何时你调用hi(), greet()和welcome()将会同时被调用。
    # 然后greet()和welcome()函数在hi()函数之外是不能访问的，比如：
    #greet()
    #outputs: NameError: name 'greet' is not defined

if __name__ == "__main__":
    common_func()
    func_in_func()