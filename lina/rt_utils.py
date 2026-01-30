import numpy as np
from astropy.io import fits
from IPython.display import clear_output, display
import subprocess
import glob
from pathlib import Path
import os
import shutil
import time
import threading

try:
    import magpyx
    from magpyx.utils import ImageStream
    import ImageStreamIOWrap as shmio
    import purepyindi
    from purepyindi import INDIClient
    import purepyindi2
    from purepyindi2 import IndiClient
except:
    print('Could not import all packages associated with XWC toolkit.')

def toggle(on, channel, client, delay=None):
    client.wait_for_properties([f'telem_{channel}.writing'])
    if delay is not None: time.sleep(delay)
    if on:
        client[f'telem_{channel}.writing.toggle'] = purepyindi.SwitchState.ON
    else:
        client[f'telem_{channel}.writing.toggle'] = purepyindi.SwitchState.OFF

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

def unpack_data(telem_path, data_path):
    subprocess.run(['xrif2fits', '-d', str(telem_path), '-D', str(data_path)])
    clear_output()

def read_telem_times(data_fnames, absolute=False):
    data_times = []
    for fname in data_fnames:
        t_hr = float(fname.split("_")[-1][8:10])
        t_min = float(fname.split("_")[-1][10:12])
        t_sec = float(fname.split("_")[-1][12:-5])/1e9
        data_times.append( 3600*t_hr + 60*t_min + t_sec )

    data_times = np.array(data_times)

    if not absolute: 
        start_time = data_times[0]
        data_times = data_times - start_time

    return data_times

def read_telem_data(data_fnames, absolute=False):
    data = []
    data_times = []
    for fname in data_fnames:
        data.append(fits.getdata(fname))
        t_hr = float(fname.split("_")[-1][8:10])
        t_min = float(fname.split("_")[-1][10:12])
        t_sec = float(fname.split("_")[-1][12:-5])/1e9
        data_times.append( 3600*t_hr + 60*t_min + t_sec )

    data = np.array(data) 
    data_times = np.array(data_times)

    if not absolute: 
        start_time = data_times[0]
        data_times = data_times - start_time

    return data, data_times

    