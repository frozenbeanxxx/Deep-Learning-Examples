import fire
from pathlib import Path

def t1():
    dirname = r'E:\temp\a1'
    tmpdir = Path(dirname)
    print(tmpdir)
    tmpdir = tmpdir / Path('aaa')
    print(tmpdir)
    print(tmpdir.exists())
    print(tmpdir.is_dir())

if __name__ == "__main__":
    fire.Fire()