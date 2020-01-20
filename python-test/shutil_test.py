import shutil
import pydoc

def t1():
    print(shutil.__doc__)
    #print(pydoc.help(shutil))
    path1 = '/media/wx/diskE/temp/sics_ocr/logs/sics_191211-145704.log'
    path2 = '/media/wx/diskE/temp/t1'
    shutil.move(path1, path2)

if __name__ == "__main__":
    t1()