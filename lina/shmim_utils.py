
import numpy as np

import threading

try:
    from magpyx.utils import ImageStream
    import ImageStreamIOWrap as shmio
except ImportError:
    print('Packages for live simulation are not available.')

def create_shmim(name, dims, dtype=None, shared=1, nbkw=8):
    # if ImageStream objects didn't auto-open on creation, you could create and return that instead. oops.
    img = shmio.Image() # not sure if I should try to destroy first in case it already exists
    buffer = np.zeros(dims)
    if dtype is None: dtype = shmio.ImageStreamIODataType.FLOAT
    img.create(name, buffer, -1, True, 8, 1, dtype, 1)

def write(STREAM, command):
    try:
        STREAM.write(np.array([command]))
    except:
        STREAM.write(np.array([command]).T)

def zero(STREAMS):
    for STREAM in STREAMS:
        try:
            STREAM.write(np.zeros(STREAM.shape))
        except:
            STREAM.write(np.zeros(STREAM.shape[::-1]))

def write_dm(STREAM, command):
    STREAM.write(1e6*np.array([command]))

def stack(
        STREAM, 
        NFRAMES=1, 
    ):

    return np.mean(STREAM.grab_many(NFRAMES), axis=0)

class Process(threading.Timer):  
    def run(self):
        while not self.finished.wait(self.interval):  
            self.function(*self.args, **self.kwargs)


    