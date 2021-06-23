print("7 classes operatetor")
print("1. +, -, *, /, **, //")
print("2. compare operator: ==, !=, >, <, >=, <=")
print("3. assignment operator: =, +=, -=, *=, /=, %=, **=, //=, := ")
print("4. bit operator : &, |, ^, ~, <<, >>")
print("5. member operater: in, not in")
print("6. id operator: is, is not")

print("\nbit operator")
print("2 << 4 = ", 2 << 4)
a = 1
print("(a := 2) == 3 is", (a:=2) == 3, ", a = ", a, ", := is 海象运算符")
print("(a := 4) == 4 is", (a:=4) == 4, ", a = ", a, ", := is 海象运算符")
l = list(range(16))
if (n := len(l)) > 10:
    print('length of l is ', n)

print("\n5. member operater")
l1 = [1, 2]
print("l1 = [1, 2], (1 in l1) = ", 1 in l1, ", (2 not in l1) = ", 2 not in l1)

