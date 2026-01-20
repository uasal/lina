import numpy as np
import astropy.units as u
from IPython.display import clear_output, display
import subprocess
import glob
from pathlib import Path
import os
import shutil
import time

import magpyx
from magpyx.utils import ImageStream
import purepyindi
from purepyindi import INDIClient
import purepyindi2
from purepyindi2 import IndiClient

def toggle(on, channel, client, delay=None):
    client.wait_for_properties([f'telem_{channel}.writing'])
    if delay is not None: time.sleep(delay)
    if on:
        client[f'telem_{channel}.writing.toggle'] = purepyindi.SwitchState.ON
    else:
        client[f'telem_{channel}.writing.toggle'] = purepyindi.SwitchState.OFF

def get_fnames(data_path):
    return sorted(glob.glob(str(data_path)))

def make_dir(dir_path):
    # Create the directory
    try:
        os.mkdir(str(dir_path))
        print(f"Directory '{str(dir_path)}' created successfully.")
    except FileExistsError:
        print(f"Directory '{str(dir_path)}' already exists.")
    except PermissionError:
        print(f"Permission denied: Unable to create '{str(dir_path)}'.")
    except Exception as e:
        print(f"An error occurred: {e}")

def move_files(source_path, target_path):
    file_names = os.listdir(str(source_path))
    for fname in file_names:
        shutil.move(str(source_path/fname), str(target_path/fname))

def delete_files(dir_path):
    fnames = sorted(glob.glob(str(dir_path)))
    for fname in fnames:
        try:
            if os.path.isfile(fname) or os.path.islink(fname):
                os.unlink(fname)
            elif os.path.isdir(fname):
                shutil.rmtree(fname)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (fname, e))

def unpack_data(telem_path, data_path):
    subprocess.run(['xrif2fits', '-d', str(telem_path), '-D', str(data_path)])
    clear_output()

from astropy.io import fits

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

    
