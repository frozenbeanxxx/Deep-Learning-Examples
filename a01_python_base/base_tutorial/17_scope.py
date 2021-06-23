num = 1
def scope_test():
    global num
    print("global num: ", num)
    num = 10
    print("local num: ", num)

def scope_test_2():
    print("\nscope_test_2")
    def outer():
        num = 10
        def inner():
            nonlocal num   # nonlocal关键字声明
            num = 100
            print(num)
        inner()
        print(num)
    outer()

if __name__ == "__main__":
    scope_test()
    scope_test_2()