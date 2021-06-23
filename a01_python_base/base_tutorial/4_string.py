import string

def string_func_startswith():
    print("\nstring_func_startswith()")
    a = 'welcome to beijing'
    b = a.startswith('wel')
    print(b)
    b = a.startswith('aaa')
    print(b)
    b = a.startswith('wel', 1, 4)
    print(b)
    b = a.startswith('elc', 1, 4)
    print(b)

    c = a.endswith('jing')
    print('end:', c)
    c = a.endswith('jing', -3, 0)
    print('end:', c)
    c = a.endswith('come', 3, 7)
    print('end:', c)

def string_func_join():
    print("\nstring_func_join()")
    s = '{} {:8.2f}'.format('clock', 1250.1234567)
    print(s)
    b = [1,2,3,4]
    s = ','.join([str(a) for a in b])
    print(s)

if __name__ == "__main__":
    print("\nString Function")
    s1 = "there is a pig"
    print(f"s1 = {s1}, s1.capitalize()", s1.capitalize())
    print("s1.center(100, '*') = ", s1.center(100, '*'))
    print("l1 = s1.split() = ", s1.split())
    l1 = s1.split()
    print("' '.join(l1) = ", ' '.join(l1))

    string_func_startswith()
    string_func_join()