def looping_statement_1():
    print("\nlooping_statement_1")
    n = 100
    sum = 0
    counter = 1
    while counter <= n:
        sum = sum + counter
        counter += 1
    print("1 到 %d 之和为: %d" % (n,sum))

def looping_statement_2():
    print("\nlooping_statement_2")
    var = 1
    while var == 1 :  # 表达式永远为 true
        num = int(input("输入一个数字  :"))
        print ("你输入的数字是: ", num)
        if (num == 666):
            break
    print ("Good bye!")

def looping_statement_3():
    print("\nlooping_statement_3")
    count = 0
    while count < 5:
        print (count, " 小于 5")
        count = count + 1
    else:
        print (count, " 大于或等于 5")

def looping_statement_4():
    print("\nlooping_statement_4")
    sites = ["Baidu", "Google","Runoob","Taobao"]
    for site in sites:
        if site == "Runoob":
            print("菜鸟教程!")
            break
        print("循环数据 " + site)
    else:
        print("没有循环数据!")
    print("完成循环!")

def range_statement():
    print("\nrange_statement")
    print("range(10) = ", range(10))
    print("list(range(10)) = ", list(range(10)))

def control_flow_statement_1():
    print("\ncontrol_flow_statement_1")
    for letter in 'Runoob':     # 第一个实例
        if letter == 'b':
            break
        print ('当前字母为 :', letter)
    var = 10                    # 第二个实例
    while var > 0:              
        print ('当期变量值为 :', var)
        var = var -1
        if var == 5:
            break
    print ("Good bye!")

def control_flow_statement_2():
    print("\ncontrol_flow_statement_2")
    for letter in 'Runoob':     # 第一个实例
        if letter == 'o':        # 字母为 o 时跳过输出
            continue
        print ('当前字母 :', letter)
 
    var = 10                    # 第二个实例
    while var > 0:              
        var = var -1
        if var == 5:             # 变量为 5 时跳过输出
            continue
        print ('当前变量值 :', var)
    print ("Good bye!")

def pass_statement():
    print("\npass_statement")
    for letter in 'Runoob': 
        if letter == 'o':
            pass
            print ('执行 pass 块')
        print ('当前字母 :', letter)
    print ("Good bye!")

if __name__ == "__main__":
    looping_statement_1()
    looping_statement_2()
    looping_statement_3()
    looping_statement_4()
    range_statement()
    control_flow_statement_1()
    control_flow_statement_2()
    pass_statement()