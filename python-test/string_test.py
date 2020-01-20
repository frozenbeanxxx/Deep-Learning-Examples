def t1():
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

def t2():
    s = '{} {:8.2f}'.format('clock', 1250.1234567)
    print(s)
    b = [1,2,3,4]
    s = ','.join([str(a) for a in b])
    print(s)

if __name__ == "__main__":
    import fire 
    fire.Fire()