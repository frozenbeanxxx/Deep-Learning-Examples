i = 666
print("i =", i, ", type(i):", type(i))

f = 888.888
print("f =", f, ", type(f):", type(f))

s = "string"
print("s =", s, ", type(i):", type(s))

c = b = i
print("continuous assignment c = b = i:", ", c =", c, ", b =", i, ", i =", i)

print("standard data type: Number, String, List, Tuple, Set, Dictionary")
print("could be modified: List, Dictionary, Set ")
print("can't be modified: Number, String, Tuple")

print("\nNumber:")
a, b, c, d = 20, 5.5, True, 4+3j
print("a, b, c, d = 20, 5.5, True, 4+3j")
print("type(): ", type(a), type(b), type(c), type(d))
print("isinstance(a, int) : ", isinstance(a, int))

del a
print("del a")
#print("a:", a)

print("\nString:")
print("numeric calculation: +, -, *, /, //, %, **")
print("r means no Escaped characters ", r'r"num\neric":', r"num\neric")
print("s=", s, "s[0]='w' is error")
print("s*3 = ", s*3)

print("\nList:")
print("init a empty list: l = []")
l1 = [ 'abcd', 786 , 2.23, 'runoob', 70.2 ]
l2 = [123, 'runoob']
print("l1 = ", l1, ", l2 = ", l2)
print("l1*3 = ", l1*3)
print("dir(l1) = ", dir(l1))
print("hasattr(l1, 'sort') = ", hasattr(l1, 'sort'))
print("getattr(l1, 'sort') = ", getattr(l1, 'sort'))
print('setattr')
print("reverse l1 = ", l1[-1::-1])

print("\nTuple")
tup1 = (1,2)
print("tup1 = ", tup1, "tup1 is a ", type(tup1))




