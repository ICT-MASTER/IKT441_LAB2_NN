import urllib.request
import time
import sys
import time
import urllib
import tarfile
import gzip
import shutil
from contextlib import closing
import urllib.request
import os.path
import os
import tarfile
import io
import os



## http://stackoverflow.com/questions/3667865/python-tarfile-progress-output
def get_file_progress_file_object_class(on_progress):
    class FileProgressFileObject(tarfile.ExFileObject):
        def read(self, size, *args):
            on_progress(self.name, self.position, self.size)
            return tarfile.ExFileObject.read(self, size, *args)
    return FileProgressFileObject

class TestFileProgressFileObject(tarfile.ExFileObject):
    def read(self, size, *args):
        on_progress(self.name, self.position, self.size)
        return tarfile.ExFileObject.read(self, size, *args)

class ProgressFileObject(io.FileIO):
    def __init__(self, path, *args, **kwargs):
        self._total_size = os.path.getsize(path)
        io.FileIO.__init__(self, path, *args, **kwargs)

    def read(self, size):
        sys.stdout.write("\rProgress: %d/%dmb" %(self.tell() / 1000000, self._total_size / 1000000))
        sys.stdout.flush()
        return io.FileIO.read(self, size)

def on_progress(filename, position, total_size):
    print("%s: %d of %s" %(filename, position, total_size))












def reporthook(count, block_size, total_size):
    global start_time
    if count == 0:
        start_time = time.time()
        return
    duration = time.time() - start_time
    if duration == 0.0:
        duration += 0.00001
    progress_size = int(count * block_size)
    speed = int(progress_size / (1024 * duration))
    percent = min(int(count*block_size*100/total_size),100)
    #percent = int(count * block_size * 100 / total_size)
    sys.stdout.write("\r...%d%%, %d MB, %d KB/s, %d seconds passed" %
                     (percent, progress_size / (1024 * 1024), speed, duration))
    sys.stdout.flush()



def download(url, path="./"):
    # Check if download dir exists
    if not os.path.isdir(path):
        os.mkdir(path)


    # Download the file from `url` and save it locally under `file_name`:
    split = urllib.parse.urlsplit(url)
    file_name = split.path.split("/")[-1]
    path = path + file_name
    if not os.path.isfile(path):
        file_name, headers = urllib.request.urlretrieve(url, path, reporthook=reporthook)
    print("\nDownload complete!")
    return path

def tar_gz(path):

    tarfile.TarFile.fileobject = get_file_progress_file_object_class(on_progress)
    tar = tarfile.open(fileobj=ProgressFileObject(path), mode='r:gz')
    tar.extractall("./Articles/")
    tar.close()

    # Maybe this isn't so amazing for you types out
    # there using *nix, os x, or (anything other than
    # windows that comes with tar and gunzip scripts).
    # However, if you find yourself on windows and
    # need to extract a tar.gz you're in for quite the
    # freeware/spyware/spamware gauntlet.

    # Python has everything you need built in!
    # Hooray for python!

def gz(path):
    out_file = path.replace(".gz","")
    with gzip.open(path, 'rb') as f:
        file_content = f.read()
        with open(out_file , "wb")as file:
            file.write(file_content)
    return out_file