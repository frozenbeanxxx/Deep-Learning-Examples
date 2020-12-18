from glob import glob

def test1():
    root_dir = 'E:/temp'
    files = glob(f'{root_dir}/a1')
    print('========= 1:\n', files)
    files = glob(f'{root_dir}/a[12]')
    print('========= 2:\n', files)
    files = glob(f'{root_dir}/a?')
    print('========= 3:\n', files)
    files = glob(root_dir + '/a[!1]*')
    print('========= 4:\n', files)
    files = glob(root_dir + '/a[1-3]')
    print('========= 5:\n', files)

def test2():
    root_dir = 'E:/prj/Minecraft/pyscript_UGATIT/dataset/face2mosaic/testA'
    files = glob(f'{root_dir}/*.*')
    print('========= 1:\n', files)

if __name__ == "__main__":
    #test1()
    test2()