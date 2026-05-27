import numpy as np
from astropy.io import fits
from IPython.display import clear_output, display
import subprocess
import glob
from pathlib import Path
import os
import shutil
import stat
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

def create_shmim(shmim_name, shape, dtype=np.float32):
    img = shmio.Image()
    buffer = np.zeros(shape, dtype=dtype)
    img.create(shmim_name, buffer)

def change_shmim_permissions(
        shmim_name,
        target_user,
        target_group='magaox-dev',
    ):
    # shmim_name = 'camscigain.im.shm'
    filename = f"/milk/shm/{shmim_name}.im.shm"

    # Get UID and GID from names (Unix-specific)
    try:
        uid = pwd.getpwnam(target_user).pw_uid
        gid = grp.getgrnam(target_group).gr_gid
    except KeyError as e:
        print(f"Error: User or group not found. {e}")
        exit()
    # Change the owner and group
    os.chown(filename, uid, gid)

    # Set specific permissions (e.g., owner read/write, group read, others no access - 0o640)
    # Use 0o prefix for octal in Python 3.x
    permissions = stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IWGRP
    # Equivalent to octal 0o640
    os.chmod(filename, permissions)

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

def toggle_telem(on, channel, client, delay=None):
    client.wait_for_properties([f'telem_{channel}.writing'])
    if delay is not None: time.sleep(delay)
    if on:
        client[f'telem_{channel}.writing.toggle'] = purepyindi.SwitchState.ON
    else:
        client[f'telem_{channel}.writing.toggle'] = purepyindi.SwitchState.OFF

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

    
import multiprocessing
import threading

class ContinuousProcess():

    def __init__(
            self,
            fun,
            args=[],
            kwargs={},
            name='UNNAMED',
        ):
        
        self.fun = fun
        self.args = args
        self.kwargs = kwargs
        self.name = name

        self.proc = multiprocessing.Process(target=self.continuous_fun)

    def continuous_fun(self):
        while True:
            self.fun(*self.args, **self.kwargs)

    def start(self):
        self.proc.start()
        print(f'{self.name} process started. PID = {self.proc.pid}',)

    def stop(self):
        self.proc.terminate()
        print(f'{self.name} process terminated.')

class TimedThread(threading.Timer):  
    def run(self):
        while not self.finished.wait(self.interval):  
            self.function(*self.args, **self.kwargs)

    def stop(self):
        self.cancel()
        print('Thread stopped.')



