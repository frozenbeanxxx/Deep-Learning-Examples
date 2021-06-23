def list_copy_test():
    '''
    list_copy_test
    '''
    print("\n Test: list copy test")
    l = list(range(16))
    print(l)
    a = l
    print(a)
    a[0] = 3
    print('a:', a)
    print('l:', l)
    b = l.copy()
    b[0] = 4
    print('b:', b)
    print('l:', l)

if __name__ == "__main__":
    list_copy_test()