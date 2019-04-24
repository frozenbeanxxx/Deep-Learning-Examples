# coding:utf-8

import os
import re
import sys
import tarfile

from six.moves import urllib

def download_test():
    URL = 'http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz'
    #dest_dir = "D:\\temp\\ccc"
    dest_dir = 'D:/dataset/cifar10/'
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    filename = URL.split('/')[-1]
    print(filename)
    filepath = os.path.join(dest_dir, filename)
    if not os.path.exists(filepath):
        def _progress(count, block_size, total_size):
            sys.stdout.write('\r>> Downloading %s %.1f%%' % (filename,
                float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()
        filepath, _ = urllib.request.urlretrieve(URL, filepath, _progress)
        print()
    statinfo = os.stat(filepath)
    print(statinfo)
    extracted_dir_path = os.path.join(dest_dir, "cifar-10-batches-bin")
    if not os.path.exists(extracted_dir_path):
        tarfile.open(filepath, 'r:gz').extractall(dest_dir)


download_test()
