def conditional_statement_1():
    print("\nconditional_statement_1")
    var1 = 100
    if var1:
        print ("1 - if 表达式条件为 true")
        print (var1)
    
    var2 = 0
    if var2:
        print ("2 - if 表达式条件为 true")
        print (var2)
    print ("Good bye!")

def conditional_statement_2():
    print("\nconditional_statement_2")
    age = int(input("请输入你家狗狗的年龄: "))
    print("")
    if age <= 0:
        print("你是在逗我吧!")
    elif age == 1:
        print("相当于 14 岁的人。")
    elif age == 2:
        print("相当于 22 岁的人。")
    elif age > 2:
        human = 22 + (age -2)*5
        print("对应人类年龄: ", human)

def conditional_statement_3():
    print("\nconditional_statement_3")
    num=int(input("输入一个数字："))
    if num%2==0:
        if num%3==0:
            print ("你输入的数字可以整除 2 和 3")
        else:
            print ("你输入的数字可以整除 2，但不能整除 3")
    else:
        if num%3==0:
            print ("你输入的数字可以整除 3，但不能整除 2")
        else:
            print  ("你输入的数字不能整除 2 和 3")

if __name__ == "__main__":
    conditional_statement_1()
    conditional_statement_2()
    conditional_statement_3()